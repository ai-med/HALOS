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
import logging
import os
import pickle

import numpy as np
import torch
from monai.data import DataLoader, partition_dataset_classes
from monai.transforms import (
    Compose,
    ResizeWithPadOrCropd,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandRotated,
    RandSpatialCropd,
    SpatialPadd,
    MapTransform
)

from config.config import HALOSArgs
from data.io import collate_multitask_fn, HALOSCache
from models.downsampling import DeepSupervisionDownsampled

log = logging.getLogger(__name__)


# transform to compute weights for Dice Loss
class ComputeWeightsd(MapTransform):
    def __init__(self, keys, n_classes):
        """
        This class will compute loss weights using median frequency balancing based on a given segmentation
        :param keys:

        """
        MapTransform.__init__(self, keys)
        self.n_classes = n_classes

    def __call__(self, x):
        d = dict(x)
        for key in self.key_iterator(d):
            ce_weights, dice_weights = estimate_weights_mfb(d[key], self.n_classes)
            d["ce_weights"] = ce_weights
            d["dice_weights"] = dice_weights
        return d


def estimate_weights_mfb(labels, n_classes):
    """
    taken from: https://github.com/abhi4ssj/quickNAT_pytorch/blob/master/utils/preprocessor.py
    calculates the weights as described in quickNAT paper: https://arxiv.org/abs/1801.04161

    :return: weights of shape (D,H,W)
    """
    # print('in estimate weights ')
    # print('label shape ', labels.shape)
    class_weights = np.zeros_like(labels, dtype=float)
    unique, counts = np.unique(labels, return_counts=True)

    # if a class doesn't exist we want to still weight it high (missing organ problem), so the network learns not to segment it
    counts_missing = np.zeros(n_classes)

    for i in np.arange(n_classes):
        # print('i ', i)
        unique_index = np.where(unique == i)
        # print('unique index ', unique_index)
        if np.isin(i, unique):
            counts_missing[i] = counts[unique_index]

    median_freq = np.median(counts)
    weights = np.zeros(n_classes)

    for i in np.arange(n_classes):
        if counts_missing[i] == 0:  # missing organ
            w = median_freq / counts.min()
        else:
            w = median_freq / counts_missing[i]
        mask = np.array(labels == i)

        class_weights += w * mask
        weights[i] = w

    try:
        grads = np.gradient(labels)
    except ValueError as e:
        print(e)
        print('label shape ', labels.shape)
    edge_weights = (grads[0] ** 2 + grads[1] ** 2 + grads[2] ** 2) > 0
    class_weights += 2 * edge_weights

    return torch.from_numpy(class_weights), torch.from_numpy(weights)


class HALOSDataPreparer:
    def __init__(self, param: HALOSArgs) -> None:
        self.config = param
        # self.seg_batch_size = self.config["seg_batch_size"]
        self.batch_size = self.config["batch_size"]
        self.val_batch_size = self.config['val_batch_size']
        self.clf_factor = self.config["clf_factor"]
        self.key = self.config["key"]
        self.seed = self.config["random_seed"]

        self._set_transforms()
        self._load_binary_labels()
        self._train_val_split()
        self._form_batch_data()

    def get_dataloader(self, cv_index):

        dataset_train = HALOSCache(seg_data=self.dict_seg_train_formed[cv_index],
                                   clf_data=self.dict_ukb_train_formed[cv_index],
                                   seg_transform=self.seg_train_transforms,
                                   clf_transform=self.ukb_train_transforms,
                                   clf_factor=self.clf_factor,
                                   seed=self.seed,
                                   cache_rate=self.config['cache_rate'])
        train_loader = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True,
                                  num_workers=0, drop_last=False, collate_fn=collate_multitask_fn)

        if self.config["val_number_pos_neg"] != [0, 0]:

            dataset_val = HALOSCache(seg_data=self.dict_seg_val_formed[cv_index],
                                     clf_data=self.dict_ukb_val_formed[cv_index],
                                     seg_transform=self.seg_val_transforms,
                                     clf_transform=self.ukb_val_transforms,
                                     clf_factor=self.clf_factor,
                                     seed=self.seed,
                                     cache_rate=self.config['cache_rate'])
            val_loader = DataLoader(dataset_val, batch_size=self.val_batch_size, shuffle=True,
                                    num_workers=0, drop_last=False, collate_fn=collate_multitask_fn)

            return train_loader, val_loader
        else:
            return train_loader, None

    def _set_transforms(self):
        self.seg_train_transforms = Compose(
            [
                LoadImaged(keys=[self.key, "annotation"]),
                ComputeWeightsd(keys='annotation', n_classes=self.config['num_classes']),  # self.n_classes
                EnsureChannelFirstd(keys=[self.key, "annotation"]),
                NormalizeIntensityd(keys=[self.key], nonzero=True, channel_wise=True),
                SpatialPadd(keys=[self.key, "annotation"], spatial_size=[268, 246, 156]),
                EnsureTyped(keys=["dice_weights"], dtype=torch.float32),

                RandRotated(keys=[self.key, "annotation"], range_x=(-0.52, 0.52), range_y=(-0.52, 0.52),
                            range_z=(-0.52, 0.52), prob=0.2),

                RandSpatialCropd(keys=[self.key, "annotation"], roi_size=self.config["enc_roi_size"], random_size=False,
                                 random_center=True),
                RandAdjustContrastd(keys=[self.key], prob=0.3, gamma=(0.7, 1.5)),

                EnsureTyped(keys=self.key, dtype=torch.float32),
                # TODO: dictionary transforms don't work on lists, therefore hardcoded datatype in DeepSupervisionDownsampled
                DeepSupervisionDownsampled(keys=["annotation"], class_number=7,
                                           ds_scales=[[1, 1, 1], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25],
                                                      [0.125, 0.125, 0.125], [0.0625, 0.0625, 0.0625]]),
            ]
        )
        self.seg_val_transforms = Compose(
            [
                LoadImaged(keys=[self.key, "annotation"]),
                ComputeWeightsd(keys='annotation', n_classes=self.config['num_classes']),  # self.n_classes
                EnsureChannelFirstd(keys=[self.key, "annotation"]),
                NormalizeIntensityd(keys=[self.key], nonzero=True, channel_wise=True),

                RandSpatialCropd(keys=[self.key, "annotation"], roi_size=self.config["enc_roi_size"], random_size=False,
                                 random_center=True),
                EnsureTyped(keys=self.key, dtype=torch.float32),

                # TODO: dictionary transforms don't work on lists, therefore hardcoded datatype in DeepSupervisionDownsampled
                DeepSupervisionDownsampled(keys=["annotation"], class_number=7,

                                           ds_scales=[[1, 1, 1], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25],
                                                      [0.125, 0.125, 0.125], [0.0625, 0.0625, 0.0625]]),

            ]
        )
        self.ukb_train_transforms = Compose(
            [
                LoadImaged(keys=[self.key]),
                EnsureChannelFirstd(keys=[self.key]),
                NormalizeIntensityd(keys=[self.key], nonzero=True, channel_wise=True),
                SpatialPadd(keys=[self.key], spatial_size=[268, 246, 156]),
                RandRotated(keys=[self.key], range_x=(-0.52, 0.52), range_y=(-0.52, 0.52),
                            range_z=(-0.52, 0.52), prob=0.2),

                ResizeWithPadOrCropd(keys=[self.key], spatial_size=self.config["enc_roi_size"]),
                RandAdjustContrastd(keys=[self.key], prob=0.3, gamma=(0.7, 1.5)),
                EnsureTyped(keys=self.key, dtype=torch.float32),

            ]
        )
        self.ukb_val_transforms = Compose(
            [
                LoadImaged(keys=[self.key]),

                EnsureChannelFirstd(keys=[self.key]),
                NormalizeIntensityd(keys=[self.key], nonzero=True, channel_wise=True),
                ResizeWithPadOrCropd(keys=[self.key], spatial_size=self.config["enc_roi_size"]),
                EnsureTyped(keys=self.key, dtype=torch.float32),

            ]
        )

    def _load_binary_labels(self):
        with open(os.path.join(self.config["path_seg_data"], "binary_label_all.pickle"), "rb") as f:
            self.dict_seg_binary_label = pickle.load(f)
            f.close()
        with open(self.config["path_ukb_label"], "rb") as f:
            self.dict_ukb_binary_label = pickle.load(f)
            f.close()

    def _train_val_split(self):
        """
        We randomly split the seg and ukb data intro train/val/test according to the parameter val_number_pos_neg
        and clf_train_val_ratio in config.
        If using cross validation, there will be correspondingly split folds.
        """
        # segmentation data
        if self.config["cross_validation"]:
            num_folds = self.config["cv_fold"]
        else:
            num_folds = 1
        self.dict_seg_train = dict.fromkeys(range(num_folds))
        self.dict_seg_val = dict.fromkeys(range(num_folds))
        ids = self.dict_seg_binary_label[0] + self.dict_seg_binary_label[1]
        labels = [0] * len(self.dict_seg_binary_label[0]) + [1] * len(self.dict_seg_binary_label[1])
        for i in range(num_folds):  # folds here just mean data is shuffled?
            seed = self.config["random_seed"] + i
            rs = np.random.RandomState(seed)
            id_0 = self.dict_seg_binary_label[0].copy()  # negative cases
            id_1 = self.dict_seg_binary_label[1].copy()  # positive cases
            rs.shuffle(id_0)
            rs.shuffle(id_1)

            ids_val = id_0[:self.config["val_number_pos_neg"][-1]] + id_1[:self.config["val_number_pos_neg"][0]]
            indices_val = [ids.index(i) for i in ids_val]
            labels_val = [labels[i] for i in indices_val]
            self.dict_seg_val[i] = dict(ids=ids_val, binary_labels=labels_val)

            ids_train = id_0[self.config["val_number_pos_neg"][-1]:] + id_1[self.config["val_number_pos_neg"][0]:]
            indices_train = [ids.index(i) for i in ids_train]
            labels_train = [labels[i] for i in indices_train]
            self.dict_seg_train[i] = dict(ids=ids_train, binary_labels=labels_train)
        # ukb data
        self.dict_ukb_train = dict.fromkeys(range(num_folds))
        self.dict_ukb_val = dict.fromkeys(range(num_folds))
        ids = self.dict_ukb_binary_label["0"] + self.dict_ukb_binary_label["1"]
        labels = [0] * len(self.dict_ukb_binary_label["0"]) + [1] * len(self.dict_ukb_binary_label["1"])
        for i in range(num_folds):
            ids_train, ids_val = partition_dataset_classes(data=ids, classes=labels, shuffle=True,
                                                           ratios=self.config["clf_train_val_ratio"],
                                                           seed=self.config["random_seed"] + i)
            indices_train = [ids.index(i) for i in ids_train]
            labels_train = [labels[i] for i in indices_train]
            indices_val = [ids.index(i) for i in ids_val]
            labels_val = [labels[i] for i in indices_val]
            self.dict_ukb_train[i] = dict(ids=ids_train, labels=labels_train)
            self.dict_ukb_val[i] = dict(ids=ids_val, labels=labels_val)

    def _form_batch_data(self):
        num_seg_training_sample = len(self.dict_seg_train[0]["ids"])
        num_ukb_ite = 1

        self.dict_seg_train_formed = dict.fromkeys(range(len(self.dict_seg_train)))
        self.dict_ukb_train_formed = dict.fromkeys(range(len(self.dict_ukb_train)))
        self.dict_seg_val_formed = dict.fromkeys(range(len(self.dict_seg_val)))
        self.dict_ukb_val_formed = dict.fromkeys(range(len(self.dict_ukb_val)))

        for f in range(len(self.dict_seg_train)):
            index = list(range(num_seg_training_sample))
            shuffled_id = [self.dict_seg_train[f]["ids"][m] for m in index]
            shuffled_label = [self.dict_seg_train[f]["binary_labels"][m] for m in index]
            formed = self._map_path(dataset="seg", ids=shuffled_id, labels=shuffled_label)
            self.dict_seg_train_formed[f] = {"ids": shuffled_id, "data": formed}
            ukb_batch_partition = partition_dataset_classes(data=self.dict_ukb_train[f]["ids"],
                                                            classes=self.dict_ukb_train[f]['labels'],
                                                            num_partitions=num_ukb_ite, shuffle=True, drop_last=False,
                                                            seed=self.config["random_seed"])

            shuffled_indices = [self.dict_ukb_train[f]["ids"].index(m) for i in ukb_batch_partition for m in i]
            fold_id = [str(self.dict_ukb_train[f]["ids"][i]) for i in shuffled_indices]
            fold_label = [self.dict_ukb_train[f]["labels"][i] for i in shuffled_indices]
            formed = self._map_path(dataset="ukb", ids=fold_id, labels=fold_label)
            self.dict_ukb_train_formed[f] = {"ids": fold_id, "data": formed}

            self.dict_seg_val_formed[f] = {"ids": self.dict_seg_val[f]["ids"],
                                           "data": self._map_path(dataset="seg",
                                                                  ids=self.dict_seg_val[f]["ids"],
                                                                  labels=self.dict_seg_val[f]["binary_labels"])}
            self.dict_ukb_val_formed[f] = {"ids": self.dict_ukb_val[f]["ids"],
                                           "data": self._map_path(dataset="ukb",
                                                                  ids=self.dict_ukb_val[f]["ids"],
                                                                  labels=self.dict_ukb_val[f]["labels"])}

    def _map_path(self, dataset, ids, labels):
        data = []
        ids = ids
        labels = labels
        if dataset == "seg":
            for i in range(len(ids)):
                file = os.path.join(self.config["path_seg_data"], 'data', str(ids[i]))
                data.append({"OPP": os.path.join(file, "mri_opp.nii.gz"),
                             "annotation": os.path.join(file, "annotation.nii.gz"),
                             "binary_label": labels[i],
                             })
        elif dataset == "ukb":
            for i in range(len(ids)):
                file = os.path.join(self.config["path_ukb_data"], str(ids[i]))
                data.append({"OPP": os.path.join(file + "_20201_2_0", "mri_opp.nii.gz"),
                             "binary_label": labels[i],
                             })
        else:
            print("Incorrect dataset option")
        return data
