# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, List, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.utils import BlendMode, PytorchPadMode, fall_back_tuple, look_up_option

__all__ = ["sliding_window_inference"]


def sliding_window_inference(
        img_only: bool,
        inputs_dict: dict,
        key: str,
        roi_size: Union[Sequence[int], int],
        sw_batch_size: int,
        predictor: Callable[..., torch.Tensor],
        overlap: float = 0.25,
        mode: Union[BlendMode, str] = BlendMode.CONSTANT,
        sigma_scale: Union[Sequence[float], float] = 0.125,
        padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
        cval: float = 0.0,
        sw_device: Union[torch.device, str, None] = None,
        device: Union[torch.device, str, None] = None,
        *args: Any,
        **kwargs: Any,
) -> torch.Tensor:
    """
    Implementation based on MONAI.
    """
    inputs = inputs_dict[key]
    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise AssertionError("overlap must be >= 0 and < 1.")

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size_ = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode).value, value=cval)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows

    # Create window-level importance map
    importance_map = compute_importance_map(
        get_valid_patch_size(image_size, roi_size), mode=mode, sigma_scale=sigma_scale, device=device
    )

    # Perform predictions
    output_image, count_map = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    _initialized = False
    for slice_g in range(0, total_slices, sw_batch_size):
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        window_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)
        if img_only:
            seg_prob = predictor(window_data, *args, **kwargs).to(device)  # batched patch segmentation
        else:
            dict_new = {}
            dict_new.update(
                {key: window_data,
                 "binary_label": inputs_dict["binary_label"].repeat(sw_batch_size).to(sw_device)}
            )
            seg_prob = predictor(dict_new, *args, **kwargs).to(device)

        if not _initialized:  # init. buffer at the first iteration
            output_classes = seg_prob.shape[1]
            output_shape = [batch_size, output_classes] + list(image_size)
            # allocate memory to store the full output and the count for overlapping parts
            output_image = torch.zeros(output_shape, dtype=torch.float32, device=device)
            count_map = torch.zeros(output_shape, dtype=torch.float32, device=device)
            _initialized = True

        # store the result in the proper location of the full output. Apply weights from importance map.
        for idx, original_idx in zip(slice_range, unravel_slice):
            output_image[original_idx] += importance_map * seg_prob[idx - slice_g]
            count_map[original_idx] += importance_map

    # account for any overlapping sections
    output_image = output_image / count_map

    final_slicing: List[slice] = []
    for sp in range(num_spatial_dims):
        slice_dim = slice(pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2])
        final_slicing.insert(0, slice_dim)
    while len(final_slicing) < len(output_image.shape):
        final_slicing.insert(0, slice(None))
    return output_image[final_slicing]


def sliding_window_inference_end2end(
        inputs_dict: dict,
        key: str,
        roi_size: Union[Sequence[int], int],
        sw_batch_size: int,
        predictor: Callable[..., torch.Tensor],
        overlap: float = 0.25,
        mode: Union[BlendMode, str] = BlendMode.CONSTANT,
        sigma_scale: Union[Sequence[float], float] = 0.125,
        padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
        cval: float = 0.0,
        sw_device: Union[torch.device, str, None] = None,
        device: Union[torch.device, str, None] = None,
        *args: Any,
        **kwargs: Any,
) -> Tuple[Any, Any]:
    """
    Implementation based on MONAI.
    """
    inputs = inputs_dict[key]
    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise AssertionError("overlap must be >= 0 and < 1.")

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size_ = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode).value, value=cval)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows

    # Create window-level importance map
    importance_map = compute_importance_map(
        get_valid_patch_size(image_size, roi_size), mode=mode, sigma_scale=sigma_scale, device=device
    )

    # Perform predictions
    output_image, count_map = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    _initialized = False
    for slice_g in range(0, total_slices, sw_batch_size):
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        window_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)

        # custom
        inputs_dict.update({key: window_data})
        if slice_g != 4:
            seg_prob, _ = predictor(inputs_dict, *args, **kwargs)
        else:
            seg_prob, feature_map = predictor(inputs_dict, *args, **kwargs)
        seg_prob = seg_prob.to(device)

        if not _initialized:  # init. buffer at the first iteration
            output_classes = seg_prob.shape[1]
            output_shape = [batch_size, output_classes] + list(image_size)
            # allocate memory to store the full output and the count for overlapping parts
            output_image = torch.zeros(output_shape, dtype=torch.float32, device=device)
            count_map = torch.zeros(output_shape, dtype=torch.float32, device=device)
            _initialized = True

        # store the result in the proper location of the full output. Apply weights from importance map.
        for idx, original_idx in zip(slice_range, unravel_slice):
            output_image[original_idx] += importance_map * seg_prob[idx - slice_g]
            count_map[original_idx] += importance_map

    # account for any overlapping sections
    output_image = output_image / count_map

    final_slicing: List[slice] = []
    for sp in range(num_spatial_dims):
        slice_dim = slice(pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2])
        final_slicing.insert(0, slice_dim)
    while len(final_slicing) < len(output_image.shape):
        final_slicing.insert(0, slice(None))
    return output_image[final_slicing], feature_map[0:1, :]


def _get_scan_interval(
        image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: float
) -> Tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    if len(image_size) != num_spatial_dims:
        raise ValueError("image coord different from spatial dims.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError("roi coord different from spatial dims.")

    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)


def sliding_window_inference_final_step1(
        inputs_dict: dict,
        key: str,
        roi_size: Union[Sequence[int], int],
        sw_batch_size: int,
        predictor: Callable[..., torch.Tensor],
        overlap: float = 0.25,
        mode: Union[BlendMode, str] = BlendMode.CONSTANT,
        sigma_scale: Union[Sequence[float], float] = 0.125,
        padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
        cval: float = 0.0,
        sw_device: Union[torch.device, str, None] = None,
        device: Union[torch.device, str, None] = None,
        *args: Any,
        **kwargs: Any,
) -> Tuple[Any, Any]:
    """
    Implementation based on MONAI.
    """
    inputs = inputs_dict[key]
    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise AssertionError("overlap must be >= 0 and < 1.")

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size_ = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode).value, value=cval)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows

    # Create window-level importance map
    importance_map = compute_importance_map(
        get_valid_patch_size(image_size, roi_size), mode=mode, sigma_scale=sigma_scale, device=device
    )

    # Perform predictions
    output_image, count_map = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    _initialized = False
    enc_outputs = []
    for slice_g in range(0, total_slices, sw_batch_size):
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        window_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)

        # custom
        inputs_dict.update({key: window_data})
        if slice_g != 4:
            enc_outputs_temp, _ = predictor(inputs_dict, *args, **kwargs)
        else:
            enc_outputs_temp, feature_maps = predictor(inputs_dict, *args, **kwargs)
        enc_outputs.append(enc_outputs_temp)

    return enc_outputs, feature_maps[0:1, :]


def sliding_window_inference_final_step2(
        inputs_dict: dict,
        key: str,
        roi_size: Union[Sequence[int], int],
        sw_batch_size: int,
        predictor: Callable[..., torch.Tensor],
        overlap: float = 0.25,
        mode: Union[BlendMode, str] = BlendMode.CONSTANT,
        sigma_scale: Union[Sequence[float], float] = 0.125,
        padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
        cval: float = 0.0,
        sw_device: Union[torch.device, str, None] = None,
        device: Union[torch.device, str, None] = None,
        *args: Any,
        **kwargs: Any,
) -> Tuple[Any, Any]:
    """
    Implementation based on MONAI.
    """
    inputs = inputs_dict[0][key]
    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise AssertionError("overlap must be >= 0 and < 1.")

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size_ = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode).value, value=cval)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows

    # Create window-level importance map
    importance_map = compute_importance_map(
        get_valid_patch_size(image_size, roi_size), mode=mode, sigma_scale=sigma_scale, device=device
    )

    # Perform predictions
    output_image, count_map = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    _initialized = False

    # custom
    num_windows = len(inputs_dict)
    for sw_batch in range(num_windows):
        inputs_dict_temp = inputs_dict[sw_batch]

    for slice_g in range(0, total_slices, sw_batch_size):
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]

        # custom
        seg_prob = predictor(inputs_dict, *args, **kwargs)
        seg_prob = seg_prob.to(device)

        if not _initialized:  # init. buffer at the first iteration
            output_classes = seg_prob.shape[1]
            output_shape = [batch_size, output_classes] + list(image_size)
            # allocate memory to store the full output and the count for overlapping parts
            output_image = torch.zeros(output_shape, dtype=torch.float32, device=device)
            count_map = torch.zeros(output_shape, dtype=torch.float32, device=device)
            _initialized = True

        # store the result in the proper location of the full output. Apply weights from importance map.
        for idx, original_idx in zip(slice_range, unravel_slice):
            output_image[original_idx] += importance_map * seg_prob[idx - slice_g]
            count_map[original_idx] += importance_map

    # account for any overlapping sections
    output_image = output_image / count_map

    final_slicing: List[slice] = []
    for sp in range(num_spatial_dims):
        slice_dim = slice(pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2])
        final_slicing.insert(0, slice_dim)
    while len(final_slicing) < len(output_image.shape):
        final_slicing.insert(0, slice(None))
    return output_image[final_slicing]


class Sliding_Window_Inferer_Final:
    """
       Implementation based on MONAI.
    """

    def __init__(self,
                 key: str,
                 inputs_shape: torch.Size,
                 roi_size: Union[Sequence[int], int],
                 sw_batch_size: int,
                 predictor: Callable[..., torch.Tensor],
                 overlap: float = 0.25,
                 mode: Union[BlendMode, str] = BlendMode.CONSTANT,
                 sigma_scale: Union[Sequence[float], float] = 0.125,
                 padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
                 cval: float = 0.0,
                 sw_device: Union[torch.device, str, None] = None,
                 device: Union[torch.device, str, None] = None):
        self.key = key
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.predictor = predictor
        self.overlap = overlap
        self.mode = mode
        self.sigma_scale = sigma_scale
        self.padding_mode = padding_mode
        self.cval = cval
        self.sw_device = sw_device
        self.device = device
        self.inputs_shape = inputs_shape
        self.num_spatial_dims = len(self.inputs_shape) - 2
        self.image_size_ = list(self.inputs_shape[2:])
        self.batch_size = self.inputs_shape[0]
        self.pad_size = []
        for k in range(len(self.inputs_shape) - 1, 1, -1):
            diff = max(roi_size[k - 2] - self.inputs_shape[k], 0)
            half = diff // 2
            self.pad_size.extend([half, diff - half])

        if self.overlap < 0 or self.overlap >= 1:
            raise AssertionError("overlap must be >= 0 and < 1.")

        self.roi_size = fall_back_tuple(self.roi_size, self.image_size_)
        self.image_size = tuple(max(self.image_size_[i], self.roi_size[i]) for i in range(self.num_spatial_dims))
        scan_interval = _get_scan_interval(self.image_size, self.roi_size, self.num_spatial_dims, self.overlap)

        self.slices = dense_patch_slices(self.image_size, self.roi_size, scan_interval)
        self.num_win = len(self.slices)  # number of windows per image
        self.total_slices = self.num_win * self.batch_size  # total number of windows

    def forward_encoder(self, inputs_dict: dict, *args: Any, **kwargs: Any):
        inputs = inputs_dict[self.key]
        inputs = F.pad(inputs, pad=self.pad_size, mode=look_up_option(self.padding_mode, PytorchPadMode).value,
                       value=self.cval)
        # Perform predictions
        enc_outputs = []
        for slice_g in range(0, self.total_slices, self.sw_batch_size):
            slice_range = range(slice_g, min(slice_g + self.sw_batch_size, self.total_slices))
            unravel_slice = [
                [slice(int(idx / self.num_win), int(idx / self.num_win) + 1), slice(None)] + list(
                    self.slices[idx % self.num_win])
                for idx in slice_range
            ]
            window_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(self.sw_device)
            # custom
            inputs_dict.update({self.key: window_data})
            if slice_g != 4:
                enc_outputs_temp, _ = self.predictor(inputs_dict, *args, **kwargs)
            else:
                enc_outputs_temp, feature_maps = self.predictor(inputs_dict, *args, **kwargs)
            enc_outputs.append(enc_outputs_temp)

        return enc_outputs, feature_maps[0:1, :]

    def forward_decoder(self, inputs_dict: dict, *args: Any, **kwargs: Any):
        importance_map = compute_importance_map(
            get_valid_patch_size(self.image_size, self.roi_size), mode=self.mode, sigma_scale=self.sigma_scale,
            device=self.device
        )

        output_image, count_map = torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
        _initialized = False

        # custom
        for slice_g in range(0, self.total_slices, self.sw_batch_size):
            slice_range = range(slice_g, min(slice_g + self.sw_batch_size, self.total_slices))
            unravel_slice = [
                [slice(int(idx / self.num_win), int(idx / self.num_win) + 1), slice(None)] + list(
                    self.slices[idx % self.num_win])
                for idx in slice_range
            ]
            inputs_dict_temp = inputs_dict[int(slice_g / 2)]
            seg_prob, _ = self.predictor(inputs_dict_temp, *args, **kwargs)
            seg_prob = seg_prob.to(self.device)

            if not _initialized:  # init. buffer at the first iteration
                output_classes = seg_prob.shape[1]
                output_shape = [self.batch_size, output_classes] + list(self.image_size)
                # allocate memory to store the full output and the count for overlapping parts
                output_image = torch.zeros(output_shape, dtype=torch.float32, device=self.device)
                count_map = torch.zeros(output_shape, dtype=torch.float32, device=self.device)
                _initialized = True

            # store the result in the proper location of the full output. Apply weights from importance map.
            for idx, original_idx in zip(slice_range, unravel_slice):
                output_image[original_idx] += importance_map * seg_prob[idx - slice_g]
                count_map[original_idx] += importance_map

        # account for any overlapping sections
        output_image = output_image / count_map

        final_slicing: List[slice] = []
        for sp in range(self.num_spatial_dims):
            slice_dim = slice(self.pad_size[sp * 2],
                              self.image_size_[self.num_spatial_dims - sp - 1] + self.pad_size[sp * 2])
            final_slicing.insert(0, slice_dim)
        while len(final_slicing) < len(output_image.shape):
            final_slicing.insert(0, slice(None))
        return output_image[final_slicing]
