# This file is part of Hallucination-free Organ Segmentation after Organ Resection Surgery (HALOS).
#
# HALOS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HALOS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HALOS. If not, see <https://www.gnu.org/licenses/>.
from monai.transforms import (
    AsDiscrete,
    MapTransform
)
from torch.nn.functional import avg_pool2d, avg_pool3d


class DeepSupervisionDownsampled(MapTransform):
    """
    Return one hot encodings of the segmentation maps if downsampling has occured (no one hot for highest resolution)
    downsampled segmentations are smooth, not 0/1
    """

    def __init__(self, keys, class_number, ds_scales):
        super().__init__(keys)
        self.class_number = class_number
        self.to_one_hot = AsDiscrete(to_onehot=self.class_number)
        self.ds_scales = ds_scales

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.downsample_seg_for_ds_transform3(d[key])
        return d

    def downsample_seg_for_ds_transform3(self, gt):
        output = []
        one_hot = self.to_one_hot(gt)

        for s in self.ds_scales:
            if all([i == 1 for i in s]):
                output.append(one_hot.astype('torch.uint8'))
            else:
                kernel_size = tuple(int(1 / i) for i in s)
                stride = kernel_size
                pad = tuple((i - 1) // 2 for i in kernel_size)

                if len(s) == 2:
                    pool_op = avg_pool2d
                elif len(s) == 3:
                    pool_op = avg_pool3d
                else:
                    raise RuntimeError()
                pooled = pool_op(one_hot, kernel_size, stride, pad, count_include_pad=False, ceil_mode=False)

                output.append(pooled.astype('torch.uint8'))

        return output
