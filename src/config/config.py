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
"""
Contains pre-defined configurations to be used for reproducible experiment.
"""
from typing import List

from typing_extensions import TypedDict

HALOSArgs = TypedDict(
    "HALOSArgs",
    {
        "path_seg_data": str,
        "path_ukb_data": str,
        "path_ukb_label": str,
        "organ": str,
        "key": str,
        "clf_ukb_weight": List[float],
        "random_seed": int,
        "device": str,
        "max_epochs": int,
        "val_interval": int,
        # unet
        "model": str,
        "enc_roi_size": List[int],
        "val_number_pos_neg": List[int],
        "seg_val_monitor_metric": str,
        # clf
        "clf_feature_loca": int,
        "clf_model": str,
        "clf_dropout": float,
        "clf_channels_conv": List[int],
        "clf_channels_dense": List[int],
        "clf_train_val_ratio": List[int],
        "clf_val_monitor_metric": str,
        # feature fusion
        "fusion_loca": int,
        "fusion_squeeze_factor": int,
        # training
        "cross_validation": bool,
        "cv_fold": int,
        "loss_weight_seg": float,
        "optimizer": str,
        "seg_lr": float,
        "clf_lr": float,
        "weight_decay": float,
        "num_classes": int,
        "clf_factor": int,
        'batch_size': int,
        'val_batch_size': int,
        'cache_rate': float,
        'clf_update': str,
        'fusion_input': str
    }
)


class HALOS_CONF:
    """
    Reproducible configurations for the HALOS model: UNet + multitask + feature fusion modules.
    """
    DEFAULT: HALOSArgs = dict()

    PARAM = DEFAULT.copy()
    PARAM.update(
        dict(
            # data
            # put the path to your datafiles here
            # path to fully annotated segmentation dataset (for training the U-Net) - check README for directory structure
            path_seg_data="/data/gallbladder/segmentation",
            # path to large scale dataset with only image labels
            path_ukb_data="/data/gallbladder/UKB/train",
            path_ukb_label="/data/gallbladder/UKB/train_binary_labels.pickle",
            save_path='experiments/halos_gallbladder',
            organ="gallbladder",
            key="OPP",
            # ground-truth class ratio in ukb dataset
            # gallbladder: [positive (without gallbladder), negative], e.g. [0.38, 0.62]
            # kidney: [positive_right (right kidney missing), positive_left, negative], e.g. [0.02, 0.23, 0.75]
            clf_ukb_weight=[0.38, 0.62],
            # fixed random seed
            random_seed=3407,
            device="cuda",
            max_epochs=1000,
            val_interval=4,
            # UNet type
            model="nnUNet",
            # input image size
            enc_roi_size=[160, 160, 96],
            # number of positive and negative data in each sampled validation set
            # gallbladder: [positive (without gallbladder), negative], e.g. [3, 7]
            # kidney: [positive_right (right kidney missing), positive_left, negative], e.g. [1, 1, 4]
            # if no need of validation, please set it to [0, 0] or [0, 0, 0]
            val_number_pos_neg=[3, 7],
            seg_val_monitor_metric="average",
            # dropout probability
            dropout_p=0,
            dropout_in_localization=False,
            # number of features in first encoder block
            base_num_features=32,

            # classifier
            clf_feature_loca=4,
            clf_model="CLFCNN",
            clf_dropout=0,
            clf_channels_conv=[256],
            # for gallbladder: clf_channels_dense=2, kidney: 3
            clf_channels_dense=[2],
            clf_train_val_ratio=[4, 1],
            clf_val_monitor_metric="Balanced_Accuracy",
            lr=1e-5,
            # feature fusion can be done at location 1(do only once at bottleneck), 2(before each level's decoder block)
            # or 3(after each decoder block)
            # default location for HALOS is fusion_loca=2
            fusion_loca=2,
            fusion_squeeze_factor=4,
            # training
            cross_validation=False,
            cv_fold=1,
            loss_weight_seg=0.5,
            optimizer="AdamW",
            seg_lr=0.000818,
            clf_lr=0.00087,
            weight_decay=0.000061,
            # number of classes to segment
            num_classes=7,
            # factor that defines how many classification samples per segmentation sample (if main_data = 'seg')
            clf_factor=2,
            batch_size=1,
            val_batch_size=2,
            # possible values: 'ukb_only' (HALOS default) = compute classification loss only on ukb data,
            # 'both' = compute classification loss on ukb and segmentation data,
            # 'none' = don't update classifier at all (only useful for debugging or training a baseline unet without classification)
            clf_update='ukb_only',
            # percentage of the data to be cached
            cache_rate=1.0,
            # should the DAFT module get the ground truth labels ('gt') or the classfication output ('clf') as input? 'gt' is default but at test time can be switched to 'clf'
            fusion_input='gt',
            # normalization, batch norm 'BN' default, other option: 'IN' for instance norm
            norm='BN'
        )
    )
