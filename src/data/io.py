import logging
import os
from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch
from monai.data import CacheDataset
from monai.transforms import (
    Compose,
)
from torch.utils.data import Dataset as _TorchDataset

log = logging.getLogger(__name__)


class DatasetBaselineSeg(CacheDataset):
    def __init__(self, mode: str, path: str, transforms: Compose, dict_data: dict) -> None:
        self.mode = mode
        self.path = path
        self.data = []
        self.files = []
        self.dict_data = dict_data
        self.map_data()
        super().__init__(data=self.data, transform=transforms, cache_rate=1.0, num_workers=4)

    def map_data(self) -> None:
        """
        Create a dictionary to map different modalities that belong to the same patient.
        """
        ids = self.dict_data["ids"]
        labels = self.dict_data["binary_labels"]

        for i in range(len(ids)):
            self.files.append(ids[i])
            file = os.path.join(self.path, self.mode, str(ids[i]))
            self.data.append({"OPP": os.path.join(file, "mri_opp.nii.gz"),
                              "annotation": os.path.join(file, "annotation.nii.gz"),
                              "binary_label": labels[i],
                              })
        log.info(f"Prepared {len(self.files)} image-label pairs for the segmentation baseline")


class DatasetBaselineSegUKB(CacheDataset):
    def __init__(self, path: str, transforms: Compose, dict_data: dict) -> None:
        self.path = path
        self.data = []
        self.files = []
        self.dict_data = dict_data
        self.map_data()
        super().__init__(data=self.data, transform=transforms, cache_rate=1.0, num_workers=4)

    def map_data(self) -> None:
        """
        Create a dictionary to map different modalities that belong to the same patient.
        """
        ids = self.dict_data["ids"]
        labels = self.dict_data["binary_labels"]

        for i in range(len(ids)):
            self.files.append(ids[i])
            file = os.path.join(self.path, str(ids[i]))
            self.data.append({ "OPP": os.path.join(file + "_20201_2_0", "mri_opp.nii.gz"),
                              "binary_label": labels[i],
                              })
        log.info(f"Prepared {len(self.files)} image-label pairs for the segmentation baseline")


class HALOSCache(_TorchDataset):
    """
    Multi Task Dataset, adapted from MONAI Dataset

    A generic dataset with a length property and an optional callable data transform
    when fetching a data sample.
    If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = dataset[1:4]`,
    for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

    For example, typical input data can be a list of dictionaries::

        [{                            {                            {
             'img': 'image1.nii.gz',      'img': 'image2.nii.gz',      'img': 'image3.nii.gz',
             'seg': 'label1.nii.gz',      'seg': 'label2.nii.gz',      'seg': 'label3.nii.gz',
             'extra': 123                 'extra': 456                 'extra': 789
         },                           },                           }]
    """

    def __init__(self, seg_data: Sequence, clf_data: Sequence, seg_transform: Optional[Callable] = None,
                 clf_transform: Optional[Callable] = None, seed: Optional[int] = None,
                 clf_factor: Optional[int] = 2, main_data: Optional[str] = 'seg',
                 cache_rate: Optional[float] = 1.0) -> None:
        """
        This Dataset implements a MultiTask dataset for joint training of segmentation and classification
        in our case the segmentation dataset is a lot smaller, so we sample more classification data per batch
        With batchsize = B, a batch is then formed by B * seg_data + B*clf_factor * clf_data
        e.g. 1 segmentation sample and 2 clf samples

        Args:
            seg_data: segmentation dataset
            clf_data: classification dataset
            seg_transform: transform for segmentation data
            clf_transform: transform for classification data
            seed: random seed for oversampling the classification data
            main_data: defines which of the 2 subdatasets (ukb or segmentation) to treat as the 'main' dataset
                        this one will define the length of the overall dataset. the other one will be randomly sampled,
                        how often is defined by clf_factor
            clf_factor: factor that defines how many classification samples per segmentation sample (if main_data = 'seg'), default = 2
                        or defines how many segmentation samples per ukb sample (if main_data = 'ukb')
        """
        self.seg_data = seg_data['data']
        self.clf_data = clf_data['data']
        self.seg_ids = seg_data['ids']
        self.clf_ids = clf_data['ids']
        self.cache_rate = cache_rate
        self.main_data = main_data
        self.seg_transform = seg_transform
        self.clf_transform = clf_transform
        self.R = np.random.RandomState(seed)
        self.clf_factor = clf_factor

        self.seg_data_cache = CacheDataset(data=self.seg_data, transform=self.seg_transform, cache_rate=self.cache_rate,
                                           as_contiguous=True, num_workers=4)
        self.clf_data_cache = CacheDataset(data=self.clf_data, transform=self.clf_transform, cache_rate=self.cache_rate,
                                           as_contiguous=True, num_workers=4)

        # super().__init__(data=self.data, transform=transforms, cache_rate=1, num_workers=4)

    def __len__(self) -> int:
        if self.main_data == 'seg':
            return len(self.seg_data)
        elif self.main_data == 'ukb':
            return len(self.clf_data)

    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        if self.main_data == 'seg':
            # generate clf_factor * random indices for classification data
            clf_index = list(self.R.randint(self.clf_data_cache.cache_num, size=self.clf_factor))
            print('clf index ', clf_index)

            seg_data = self.seg_data_cache[index]
            # returns subset, maybe is faster but have to adapt collate function
            # clf_data = self.clf_data_cache[clf_index]
            clf_data = [self.clf_data_cache[c] for c in clf_index]
        elif self.main_data == 'ukb':
            # TODO: so far it is not possible to sample less segmentation data than ukb data
            # generate clf_factor * random indices for segmentation data
            seg_index = list(self.R.randint(self.seg_data_cache.cache_num, size=self.clf_factor))
            print('seg_index  ', seg_index)

            clf_data = self.clf_data[index]
            # returns subset, maybe is faster but have to adapt collate function
            # clf_data = self.clf_data_cache[clf_index]
            seg_data = [self.seg_data[c] for c in seg_index]
        return seg_data, clf_data


def collate_multitask_fn(batch_list):
    # seg batch might be a tuple or single dict, depending on the transforms
    if isinstance(batch_list[0][0], list):
        seg_batch = {key: [] for key in batch_list[0][0][0].keys()}
        # TODO: rest of collate won't work in this case yet

    else:
        seg_batch = {key: [] for key in batch_list[0][0].keys()}
    clf_batch = {key: [] for key in batch_list[0][1][0].keys()}

    # stack for images
    for key in ['F', 'OPP', 'W', 'IN', 'annotation', 'ce_weights', 'dice_weights', 'binary_label', 'OPP_meta_dict',
                'annotation_meta_dict']:
        if key in seg_batch:
            if key == 'annotation':
                # batch_list[0][0]['annotation'], batch_list[1][0]['annotation']
                flat_list = [item[0][key] for item in batch_list]
                new_list = []
                for i in range(len(flat_list[0])):  # loop over downsampled items
                    new_list.append(torch.stack([el[i] for el in flat_list], dim=0))
                seg_batch[key] = new_list
            elif isinstance(batch_list[0][0][key], torch.Tensor):
                # some of OPP, F etc are just strings
                seg_batch[key] = torch.stack([data[0][key] for data in batch_list], dim=0)
            elif isinstance(batch_list[0][0][key], int):  # binary labels
                seg_batch[key] = torch.stack([torch.as_tensor(data[0][key]) for data in batch_list])
            else:  # strings
                seg_batch[key] = [data[0][key] for data in batch_list]

        if key in clf_batch:
            if isinstance(batch_list[0][1][0][key], torch.Tensor):
                clf_batch[key] = torch.stack([data[key] for el in batch_list for data in el[1]], dim=0)
            elif isinstance(batch_list[0][1][0][key], int):  # binary labels
                clf_batch[key] = torch.stack([torch.as_tensor(data[key]) for el in batch_list for data in el[1]])
            else:
                clf_batch[key] = [data[key] for el in batch_list for data in el[1]]

    return seg_batch, clf_batch


def collate_multitask_fn_clf(batch_list):
    seg_batch = {key: [] for key in batch_list[0][1][0].keys()}
    clf_batch = {key: [] for key in batch_list[0][0].keys()}

    # stack for images
    for key in ['F', 'OPP', 'W', 'IN', 'annotation', 'ce_weights', 'dice_weights', 'binary_label', 'OPP_meta_dict',
                'annotation_meta_dict']:
        if key in seg_batch:
            if key == 'annotation':
                # batch_list[0][0]['annotation'], batch_list[1][0]['annotation']
                flat_list = [item[0][key] for item in batch_list]
                new_list = []
                for i in range(len(flat_list[0])):  # loop over downsampled items
                    new_list.append(torch.stack([el[i] for el in flat_list], dim=0))
                seg_batch[key] = new_list
            elif isinstance(batch_list[0][0][key], torch.Tensor):
                # some of OPP, F etc are just strings
                seg_batch[key] = torch.stack([data[0][key] for data in batch_list], dim=0)
            elif isinstance(batch_list[0][0][key], int):  # binary labels
                seg_batch[key] = torch.stack([torch.as_tensor(data[0][key]) for data in batch_list])
            else:  # strings
                seg_batch[key] = [data[0][key] for data in batch_list]

        if key in clf_batch:
            if isinstance(batch_list[0][1][0][key], torch.Tensor):
                clf_batch[key] = torch.stack([data[key] for el in batch_list for data in el[1]], dim=0)
            elif isinstance(batch_list[0][1][0][key], int):  # binary labels
                clf_batch[key] = torch.stack([torch.as_tensor(data[key]) for el in batch_list for data in el[1]])
            else:
                clf_batch[key] = [data[key] for el in batch_list for data in el[1]]

    return seg_batch, clf_batch
