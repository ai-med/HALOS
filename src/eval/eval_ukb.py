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
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.join(Path(__file__).parent.parent.parent))
import wandb
from data.io import DatasetBaselineSegUKB
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    KeepLargestConnectedComponent,
    SaveImage,
    NormalizeIntensityd,
    EnsureTyped,
    AsDiscrete
)
from models.modified_UNet import Final_UNet
from models.classifier import CLFCNN
from models.metrics import CLFMetric, GallbladderOnly, KidneyOnly_separate
from utils import utils
from models.initialization import InitWeights_He
from monai.data import DataLoader, decollate_batch
from models.inferer import Sliding_Window_Inferer_Final

wandb.login()


def eval_fold():
    global f
    path_current = os.path.join(path_run, "fold_" + str(cv_fold))
    for checkpoint in checkpoint_list:
        # update config
        if checkpoint == "alone":
            unet_name = "fused_unet_best_alone.pth"
            clf_name = "clf_best_alone.pth"
        elif checkpoint == "common":
            unet_name = "fused_unet_best_common.pth"
            clf_name = "clf_best_common.pth"
        elif checkpoint == "last":
            unet_name = "fused_unet_final_checkpoint.pth"
            clf_name = "clf_final_checkpoint.pth"
        run_wandb = wandb.init(project=project_name, name=run + "_ukb_" + checkpoint, config=config,
                               tags=["fold_" + str(cv_fold) + "_" + checkpoint],
                               entity=entity, dir=path_current, reinit=True)

        if organ == "kidney":
            clf_metric_test = CLFMetric(num_classes=3)
            fpr_metric_test = KidneyOnly_separate(save_individual_result=True)
        else:
            clf_metric_test = CLFMetric(num_classes=2)
            fpr_metric_test = GallbladderOnly(save_individual_result=True)
        dataset_test = DatasetBaselineSegUKB(path=path_data, transforms=test_transforms, dict_data=dict_gt)
        loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        unet_dict = torch.load(os.path.join(path_current, unet_name), map_location=device)
        unet.load_state_dict(unet_dict)
        if clfNet is not None:
            clf_dict = torch.load(os.path.join(path_current, clf_name), map_location=device)
            clfNet.load_state_dict(clf_dict)
            clfNet.eval()

        unet.eval()
        fpr_metric_test.reset()
        clf_metric_test.reset()
        clf_preds = {}
        with torch.no_grad():
            for test_data in loader_test:
                test_inputs, test_clf_labels = (
                    test_data[key].to(device),
                    test_data["binary_label"].to(device)
                )
                test_inputs_dict = {"encoder": True, key: test_inputs, "ukb_only": False}
                test_enc_outputs, test_feature_maps = inferer.forward_encoder(inputs_dict=test_inputs_dict)
                if clfNet is not None:
                    test_clf_outputs = clfNet(test_feature_maps)

                    softmax_clf_outputs = softmax(test_clf_outputs)
                if config['fusion_method'] == "none":
                    _ = [k.update({"encoder": False}) for k in test_enc_outputs]
                else:
                    if use_real_pred:
                        if organ == "kidney":
                            _ = [k.update({"encoder": False,
                                           "clf_results": softmax_clf_outputs.repeat(sw_batch_size, 1)})
                                 for k in test_enc_outputs]
                        else:
                            _ = [k.update({"encoder": False,
                                           "clf_results": softmax_clf_outputs[:, -1].repeat(sw_batch_size)})
                                 for k in test_enc_outputs]
                    else:
                        _ = [k.update(
                            {"encoder": False, "clf_results": test_clf_labels.repeat(sw_batch_size)}) for k
                            in test_enc_outputs]
                test_outputs = inferer.forward_decoder(inputs_dict=test_enc_outputs)
                preds_list = decollate_batch(test_outputs)
                preds_convert = [post_transforms(k) for k in preds_list]
                img_id = test_data[key + "_meta_dict"]["filename_or_obj"][0].split("/")[-2].split("_")[0]
                fpr_metric_test.compute_metrics_eval(preds_convert=preds_convert, binary_label=test_clf_labels,
                                                     img_id=img_id)
                if clfNet is not None:
                    clf_metric_test.update(pred=test_clf_outputs.cpu(), gt=test_clf_labels.cpu())
                    clf_results = int(softmax_clf_outputs.argmax(dim=1).cpu().numpy())
                    clf_preds.update({img_id: clf_results})
                if save_predictions:
                    preds_convert = preds_convert[0].cpu().numpy()
                    result = []
                    result.append(0 * (preds_convert[0]))
                    result.append(1 * (preds_convert[1]))
                    result.append(2 * (preds_convert[2]))
                    result.append(3 * (preds_convert[3]))
                    result.append(4 * (preds_convert[4]))
                    result.append(5 * (preds_convert[5]))
                    result.append(6 * (preds_convert[6]))
                    result = np.stack(result, axis=0).astype(np.float32)
                    preds_convert = np.sum(result, axis=0, keepdims=True)
                    pred_saver = SaveImage(output_dir=os.path.join(path_current, "test", checkpoint, tag),
                                           output_postfix="pred_" + img_id, output_ext=".nii.gz",
                                           separate_folder=False)
                    pred_saver(preds_convert)
            fpr_metric_test.update_metrics()
            fpr_results = fpr_metric_test.results
            if clfNet is not None:
                clf_metrics = clf_metric_test.calculate_metrics()
                utils.record_metric_results_wandb(run=run_wandb, data="ukbtest_" + tag, dict_metrics=clf_metrics,
                                                  epoch=0)
            utils.record_metric_results_wandb(run=run_wandb, data="ukbtest_" + tag, dict_metrics=fpr_results,
                                              epoch=0)

        Path(os.path.join(path_current, "test_UKB", checkpoint, tag)).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(path_current, "test_UKB", checkpoint, tag, "test_UKB_results.pkl"), "wb") as f:
            save_clf = None if clfNet is None else clf_metrics

            history = dict(average=fpr_metric_test.results, individual=fpr_metric_test.results_individual,
                           clf=save_clf, clf_preds=clf_preds)
            pickle.dump(history, f)
            f.close()


if __name__ == "__main__":
    # specify the run folder
    list_runs = ["/experiments/halos_kidney/20221207_203752"]
    all_folds = True
    specific_fold = 0
    # specify the checkpoint name
    checkpoint_list = ["alone, last"]  # alone, common, last
    save_predictions = False
    use_real_pred = False

    path_data = "/data/kidney/UKB/test/"
    device = torch.device("cuda")
    project_name = "HALOS_results"
    entity = None

    path_dict_gt = "/data/kidney/UKB/dict_kidney_test.pickle"
    with open(path_dict_gt, "rb") as f:
        dict_gt = pickle.load(f)
        f.close()
    if use_real_pred:
        tag = "use_pred"
    else:
        tag = "use_gt"

    for path_run in list_runs:
        run = path_run.split("/")[-1]
        with open(os.path.join(path_run, "config.pkl"), "rb") as f:
            config = pickle.load(f)
            f.close()
        key = config["key"]
        organ = config["organ"]
        model = config["model"]
        roi_size = config["enc_roi_size"]
        dropout_p = config["dropout_p"]
        dropout_in_localization = config["dropout_in_localization"]
        base_num_features = config["base_num_features"]
        clf_feature_loca = config["clf_feature_loca"]
        clf_dropout = config["clf_dropout"]
        clf_channels_conv = config["clf_channels_conv"]
        clf_channels_dense = config["clf_channels_dense"]
        fusion_loca = config["fusion_loca"]
        fusion_squeeze_factor = config["fusion_squeeze_factor"]
        fusion_feature_dim = config["fusion_feature_dim"]
        num_classes = config["num_classes"]
        norm = config["norm"]
        if organ == "kidney":
            if config['fusion_input'] == 'clf':
                fusion_feature_dim = 3
            else:
                fusion_feature_dim = 1
        else:
            fusion_feature_dim = 1
        # model initialization
        if config['norm'] == 'BN':
            norm_op = nn.BatchNorm3d
        else:
            norm_op = nn.InstanceNorm3d
        if config["model"] == "nnUNet":
            unet = Final_UNet(
                input_channels=1, base_num_features=config['base_num_features'], num_classes=7, num_pool=5,
                num_conv_per_stage=2,
                feat_map_mul_on_downscale=2, conv_op=nn.Conv3d,
                norm_op=norm_op, norm_op_kwargs={'eps': 1e-5, 'affine': True},
                dropout_op=nn.Dropout3d, dropout_op_kwargs={'p': 0, 'inplace': True},
                nonlin=nn.LeakyReLU, nonlin_kwargs={'negative_slope': 1e-2, 'inplace': True},
                deep_supervision=True, dropout_in_localization=False, final_nonlin=lambda x: x,
                weightInitializer=InitWeights_He(1e-2),
                pool_op_kernel_sizes=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
                conv_kernel_sizes=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                upscale_logits=False, convolutional_pooling=True, convolutional_upsampling=True,
                clf_feature_loca=clf_feature_loca,
                fusion_loca=fusion_loca, fusion_squeeze_factor=fusion_squeeze_factor,
                fusion_feature_dim=fusion_feature_dim,
            ).to(device)
        b = config['base_num_features']
        list_clf_input_channel = [b, b * 2, b * 4, b * 8, min(320, b * 16), min(320, b * 32), min(320, b * 16), b * 8,
                                  b * 4, b * 2, b]
        clfNet = CLFCNN(in_channels=list_clf_input_channel[clf_feature_loca - 1],
                        channels_conv=config["clf_channels_conv"], channels_dense=config["clf_channels_dense"],
                        dropout=config["clf_dropout"], norm_op=norm_op).to(device)

        test_transforms = Compose(
            [
                LoadImaged(keys=[key]),
                EnsureChannelFirstd(keys=[key]),
                NormalizeIntensityd(keys=[key], nonzero=True, channel_wise=True),
                EnsureTyped(keys=key, dtype=torch.float32),
            ]
        )
        post_transforms = Compose([AsDiscrete(argmax=True, to_onehot=7),
                                   KeepLargestConnectedComponent(applied_labels=[1, 2, 3, 4, 6])])
        softmax = nn.Softmax(dim=1)
        sw_batch_size = 2
        inferer = Sliding_Window_Inferer_Final(key=key, inputs_shape=torch.Size([1, 1, 192, 176, 112]),
                                               roi_size=roi_size, sw_batch_size=sw_batch_size, predictor=unet,
                                               overlap=0.5, mode="gaussian", device=device, sw_device=device)
        if all_folds is not None:
            for cv_fold in range(5):
                eval_fold()
        elif specific_fold is not None:
            cv_fold = specific_fold
            eval_fold()
