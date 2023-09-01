"""
HALOS model: Multitask training of a baseline UNet and CLF layers with feature fusion modules (DAFT).
Use both segmentation and UKB dataset.
Target organ: kidneys.
"""
import logging
import os
import pickle
import random
import sys
import warnings
from importlib import reload
from pathlib import Path

import numpy as np
import torch
import wandb
from torch import nn

wandb.login()

sys.path.insert(0, os.path.join(Path(__file__).parent.parent.parent))
from monai.utils import set_determinism

from utils import utils
from utils.parser import parse_config_halos
from config.config import HALOS_CONF
from data.preprocessing.pipeline import HALOSDataPreparer

from models.modified_UNet import Final_UNet
from models.classifier import CLFCNN
from models.initialization import InitWeights_He
from models.metrics import SegMetric, CLFMetric, KidneyOnly
from models.losses import MultipleOutputLoss, WeightedDiceCELoss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_fold(config, cv_index, path_run, dataPreparer):
    # load and update config
    project_name = "halos_experiments"
    entity = None
    path_current = os.path.join(path_run, "fold_" + str(cv_index))
    Path(path_current).mkdir(parents=True, exist_ok=True)
    logging.shutdown()
    reload(logging)
    logging.basicConfig(filename=os.path.join(path_current, "run.log"),
                        level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    log = logging.getLogger(__name__)

    # naming of the experiment. This contains all important hyperparameters, feel free to change
    prefix = "HALOS_gallbladder_" + str(config["fusion_loca"]) + "_loca" + str(
        config["clf_feature_loca"]) + "_clf" + str(config["clf_channels_conv"][0]) + "_alpha" + str(
        config["loss_weight_seg"]) + "_seg_lr" + str(config["seg_lr"]) + "_clf_lr" + str(
        config["clf_lr"]) + "_decay" + str(config["weight_decay"]) + "_clfupdate_" + str(
        config['clf_update']) + '_bs_' + str(config['batch_size']) + '_clf_factor_' + str(
        config['clf_factor']) + '_'

    run = wandb.init(project=project_name, name=prefix + path_run.split("/")[-1], config=config,
                     tags=["all_fold_" + str(cv_index)], entity=entity, dir=path_current, reinit=True,
                     settings=wandb.Settings(start_method="fork"))

    device = torch.device(config["device"])
    key = config["key"]
    max_epochs = config["max_epochs"]
    val_interval = config["val_interval"]
    clf_feature_loca = config["clf_feature_loca"]
    fusion_loca = config["fusion_loca"]
    fusion_squeeze_factor = config["fusion_squeeze_factor"]
    seg_lr = config["seg_lr"]
    clf_lr = config["clf_lr"]

    weight_decay = config["weight_decay"]
    seg_val_monitor_metric = config["seg_val_monitor_metric"]
    clf_val_monitor_metric = config["clf_val_monitor_metric"]
    avg_val_monitor_metric = lambda dice, bacc: 0.5 * (dice + bacc)
    loss_weight_seg = config["loss_weight_seg"]

    if config['fusion_input'] == 'clf':
        fusion_feature_dim = 3
    else:
        fusion_feature_dim = 1

    loader_train, loader_val = dataPreparer.get_dataloader(cv_index=cv_index)
    set_determinism(seed=config["random_seed"])

    # model initialization
    if config['norm'] == 'BN':
        norm_op = nn.BatchNorm3d
    else:
        norm_op = nn.InstanceNorm3d

    # create u-net model
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
        fusion_loca=fusion_loca, fusion_squeeze_factor=fusion_squeeze_factor, fusion_feature_dim=fusion_feature_dim,
    ).to(device)

    # based on the location of the classifier, the classifier will get a certain amount of features
    # for a U-net starting with b channels: (in our case the initial channels in the U-Net are 32, but the parameter can be reduced to save some memory)
    # example for base_num_feat 32:
    # list_clf_input_channel = [32, 64, 128, 256, 320, 320, 320, 256, 128, 64, 32]
    b = config['base_num_features']
    list_clf_input_channel = [b, b * 2, b * 4, b * 8, min(320, b * 16), min(320, b * 32), min(320, b * 16), b * 8,
                              b * 4, b * 2, b]

    # create the classifier branch:
    clfNet = CLFCNN(in_channels=list_clf_input_channel[clf_feature_loca - 1],
                    channels_conv=config["clf_channels_conv"], channels_dense=config["clf_channels_dense"],
                    dropout=config["clf_dropout"], norm_op=norm_op).to(device)

    # print number of network parameters:
    print("U-Net params ", count_parameters(unet))

    # define loss
    seg_loss_function = WeightedDiceCELoss(include_background=True, to_onehot_y=False, softmax=True, smooth_dr=1,
                                           smooth_nr=1, ce_weight=None, reduction="mean", batch=True)

    seg_loss_wrapper = MultipleOutputLoss(loss=seg_loss_function,
                                          weight_factors=[0.53333333, 0.26666667, 0.13333333, 0.06666667, 0.])

    # these weights balance the class imbalance of our UKBiobank dataset (weakly annotated data)
    # in our experiments the class 'gallbladder exists' was more prevalent. weights are the inverse frequency
    # adapt these to your individual dataset
    clf_ukb_weight = torch.Tensor(config["clf_ukb_weight"]).to(device)
    clf_ukb_loss_function = torch.nn.CrossEntropyLoss(weight=clf_ukb_weight, reduction="mean")
    weighted_loss = lambda seg, clf: loss_weight_seg * seg + (1 - loss_weight_seg) * clf

    # define optimizer, we used AdamW per default in our experiments
    if config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(
            [{'params': unet.parameters()},
             {'params': clfNet.parameters(), 'lr': clf_lr}],
            lr=seg_lr, weight_decay=weight_decay, momentum=0.99, nesterov=True)

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                         lr_lambda=lambda epoch: (1 - epoch / max_epochs) ** 0.9)
    elif config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            [{'params': unet.parameters()},
             {'params': clfNet.parameters(), 'lr': clf_lr}],
            lr=seg_lr, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    elif config["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(
            [{'params': unet.parameters()},
             {'params': clfNet.parameters(), 'lr': clf_lr}],
            lr=seg_lr, weight_decay=weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    # set metrics
    seg_metric_train = SegMetric(organ="kidney", mode="train", save_individual_result=False)
    seg_metric_val = SegMetric(organ="kidney", mode="train", save_individual_result=False)
    fpr_metric_val = KidneyOnly(save_individual_result=False)
    clf_metric_train = CLFMetric(num_classes=3)
    clf_metric_val = CLFMetric(num_classes=3)

    softmax = nn.Softmax(dim=1)

    best_seg_metric = -1
    best_clf_metric = -1
    best_avg_metric = -1
    avg_val_results = []

    scaler = torch.cuda.amp.GradScaler()
    wandb.watch(unet, log='all', log_freq=10, log_graph=True)

    for epoch in range(max_epochs):
        print('epoch ', epoch)
        # training
        unet.train()
        clfNet.train()
        epoch_clf_loss = 0
        epoch_seg_loss = 0
        step = 0

        for counter, train_batch_data in enumerate(loader_train):
            step += 1
            num_seg_img = train_batch_data[0][key].shape[0]

            # concatenate labeled and weakly labeled images, for forward pass of the encoder
            inputs = torch.cat([train_batch_data[0][key], train_batch_data[1][key]], dim=0).contiguous().to(device)
            seg_labels = [label.to(device) for label in train_batch_data[0]["annotation"]]
            clf_labels = torch.cat([train_batch_data[0]["binary_label"], train_batch_data[1]["binary_label"]],
                                   dim=0).to(device)
            seg_dice_weights = train_batch_data[0]["dice_weights"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                # common forward pass of encoder:
                inputs_dict = {"encoder": True, key: inputs, "ukb_only": False}
                enc_outputs, clf_feature_maps = unet(inputs_dict)

                clf_outputs = clfNet(clf_feature_maps)

                # due to lazy modules have to do this after first forward pass:
                if epoch == 0 and counter == 0:
                    wandb.watch(clfNet, log='all', log_freq=10)
                    print("CLF params ", count_parameters(clfNet))

                # it is also possible to train the DAFT module with the classifier output, however we use fusion_input=gt as default
                if config['fusion_input'] == 'clf':
                    # enc_outputs.update({"encoder": False, "clf_results": softmax(clf_outputs)[:, -1]})
                    enc_outputs.update({"encoder": False, "clf_results": softmax(clf_outputs[:num_seg_img]),
                                        'x': enc_outputs['x'][:num_seg_img],
                                        'skips': [s[:num_seg_img] for s in enc_outputs['skips']]})

                else:
                    # use the gt labels instead
                    enc_outputs.update({"encoder": False, "clf_results": clf_labels[:num_seg_img],
                                        'x': enc_outputs['x'][:num_seg_img],
                                        'skips': [s[:num_seg_img] for s in enc_outputs['skips']]})

                seg_outputs = unet(enc_outputs)

                # backpropagation

                seg_loss = seg_loss_wrapper(list(seg_outputs), seg_labels, loss_weights=seg_dice_weights)

                print('seg loss ', seg_loss)

                # update classifier with ukb data only or with all data:

                if config['clf_update'] == 'none':
                    # classifier won't get updated at all, this can be useful for debugging the u-net
                    loss = seg_loss

                else:
                    if config['clf_update'] == 'ukb_only':
                        # only compute the loss on classification outputs of ukb data
                        # default mode for HALOS
                        clf_loss = clf_ukb_loss_function(clf_outputs[num_seg_img:], clf_labels[num_seg_img:])
                    elif config['clf_update'] == 'both':
                        # compute the loss on classification outputs of the ukb data (weakly annotated, large dataset) and the small segmentation data
                        # in our experiments this did not work well, as the classifier will overfit on the small segmentation dataset
                        clf_loss = clf_ukb_loss_function(clf_outputs, clf_labels)

                    print('clf loss ', clf_loss)
                    loss = weighted_loss(seg_loss, clf_loss)

            # for mixed precision
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=2.0, norm_type=2)
            torch.nn.utils.clip_grad_norm_(clfNet.parameters(), max_norm=2.0, norm_type=2)

            retval = scaler.step(optimizer)
            if retval is None:
                warnings.warn('nans or inf encountered in gradients, no update done.')

            scaler.update()

            # compute training metrics (quite slow) - remove or just compute on single batch:
            with torch.no_grad():

                epoch_seg_loss += seg_loss
                if config['clf_update'] in ['both', 'ukb_only']:
                    epoch_clf_loss += clf_loss
                # computing metrics is really slow, to save time only compute it for last batch of each epoch
                # for debugging only for first
                if counter == len(loader_train) - 1:
                    seg_metric_train.compute_metrics(preds=seg_outputs, label=seg_labels,
                                                     binary_label=train_batch_data[0]["binary_label"], img_id=None)

                clf_metric_train.update(pred=clf_outputs[num_seg_img:], gt=clf_labels[num_seg_img:])

        epoch_seg_loss = epoch_seg_loss.cpu().numpy() / step
        if config['clf_update'] in ['both', 'ukb_only']:
            epoch_clf_loss = epoch_clf_loss.cpu().numpy() / step

        epoch_loss = weighted_loss(epoch_seg_loss, epoch_clf_loss)
        lr_scheduler.step()
        seg_metric_train.update_metrics()
        seg_train_epoch_results = seg_metric_train.results
        clf_train_epoch_metrics = clf_metric_train.calculate_metrics()
        run.log({'seg_lr': optimizer.param_groups[0]['lr']})
        run.log({'clf_lr': optimizer.param_groups[1]['lr']})

        run.log({"Loss/train": epoch_loss, "epoch": epoch + 1})
        run.log({"Loss_seg/train": epoch_seg_loss, "epoch": epoch + 1})
        if config['clf_update'] in ['both', 'ukb_only']:
            run.log({"Loss_clf/train": epoch_clf_loss, "epoch": epoch + 1})
        utils.record_metric_results_wandb(run=run, data="train", dict_metrics=seg_train_epoch_results, epoch=epoch + 1)
        utils.create_metric_bar_chart_wandb(run=run, metric_name="dice", dataset="train",
                                            dict_metrics=seg_train_epoch_results)
        utils.record_metric_results_wandb(run=run, data="train", dict_metrics=clf_train_epoch_metrics, epoch=epoch + 1)
        seg_metric_train.reset()
        clf_metric_train.reset()

        # validation
        if (epoch + 1) % val_interval == 0:
            unet.eval()
            clfNet.eval()
            val_epoch_seg_loss = 0
            val_epoch_clf_loss = 0

            val_step_seg = 0
            val_step_ukb = 0
            with torch.cuda.amp.autocast():
                with torch.no_grad():

                    for val_counter, val_batch_data in enumerate(loader_val):
                        num_seg_img = len(val_batch_data[0]["binary_label"])

                        val_step_seg += 1
                        val_step_ukb += 1

                        val_inputs = torch.cat([val_batch_data[0][key], val_batch_data[1][key]], dim=0).contiguous().to(
                            device)
                        val_seg_labels = [label.to(device) for label in val_batch_data[0]["annotation"]]
                        val_clf_labels = torch.cat(
                            [val_batch_data[0]["binary_label"], val_batch_data[1]["binary_label"]],
                            dim=0).to(device)
                        val_seg_dice_weights = val_batch_data[0]["dice_weights"].to(device)

                        # common forward pass of encoder:
                        val_inputs_dict = {"encoder": True, key: val_inputs, "ukb_only": False}
                        val_enc_outputs, val_feature_maps = unet(val_inputs_dict)

                        val_clf_outputs = clfNet(val_feature_maps)
                        val_enc_outputs.update({"encoder": False, "clf_results": softmax(val_clf_outputs)})
                        val_seg_outputs = unet(val_enc_outputs)

                        val_seg_outputs_seg = [s[:num_seg_img] for s in val_seg_outputs]
                        val_seg_outputs_clf = [s[num_seg_img:] for s in val_seg_outputs]

                        val_seg_loss = seg_loss_wrapper(list(val_seg_outputs_seg), val_seg_labels,
                                                        loss_weights=val_seg_dice_weights)
                        print('val seg loss ', val_seg_loss)
                        val_epoch_seg_loss += val_seg_loss

                        # update classifier with ukb data only or with all data:
                        if config['clf_update'] in ['both', 'ukb_only']:

                            if config['clf_update'] == 'both':
                                val_clf_loss = clf_ukb_loss_function(val_clf_outputs, val_clf_labels)
                            else:
                                # ukb only:
                                val_clf_loss = clf_ukb_loss_function(val_clf_outputs[num_seg_img:],
                                                                     val_clf_labels[num_seg_img:])

                            print('val clf loss ', val_clf_loss)

                            val_epoch_clf_loss += val_clf_loss
                            clf_metric_val.update(pred=val_clf_outputs[num_seg_img:], gt=val_clf_labels[num_seg_img:])

                        fpr_metric_val.compute_metrics(preds_convert=val_seg_outputs_clf,
                                                       binary_label=val_clf_labels[num_seg_img:],
                                                       img_id=None)

                        seg_metric_val.compute_metrics(preds=val_seg_outputs_seg, label=val_seg_labels,
                                                       binary_label=val_batch_data[0]["binary_label"], img_id=None)

            val_epoch_seg_loss = val_epoch_seg_loss.cpu().numpy() / val_step_seg
            if config['clf_update'] in ['both', 'ukb_only']:
                val_epoch_clf_loss = val_epoch_clf_loss.cpu().numpy() / val_step_ukb
            val_epoch_loss = weighted_loss(val_epoch_seg_loss, val_epoch_clf_loss)
            seg_metric_val.update_metrics()
            seg_val_epoch_results = seg_metric_val.results
            if config['clf_update'] in ['both', 'ukb_only']:

                clf_val_epoch_metrics = clf_metric_val.calculate_metrics()
            else:
                clf_val_epoch_metrics = None

            fpr_metric_val.update_metrics()
            fpr_val_epoch_results = fpr_metric_val.results

            run.log({"Loss/val": val_epoch_loss, "epoch": epoch + 1})
            run.log({"Loss_seg/val": val_epoch_seg_loss, "epoch": epoch + 1})
            if config['clf_update'] in ['both', 'ukb_only']:
                run.log({"Loss_clf/val": val_epoch_clf_loss, "epoch": epoch + 1})
            utils.record_metric_results_wandb(run=run, data="val", dict_metrics=seg_val_epoch_results,
                                              epoch=epoch + 1)
            utils.record_metric_results_wandb(run=run, data="val", dict_metrics=fpr_val_epoch_results,
                                              epoch=epoch + 1)
            utils.create_metric_bar_chart_wandb(run=run, metric_name="dice", dataset="val",
                                                dict_metrics=seg_val_epoch_results)
            if config['clf_update'] in ['both', 'ukb_only']:
                utils.record_metric_results_wandb(run=run, data="val", dict_metrics=clf_val_epoch_metrics,
                                                  epoch=epoch + 1)

            # saving of models based on validation metrics.
            # we save a few different models unet_best_alone = model with best segmentation metric,
            # clf_best_alone: model with best classification metric
            # unet_best_common and clf_best_common for best average metric (this is the default one used in HALOS)
            if seg_val_epoch_results[seg_val_monitor_metric][-1] > best_seg_metric:
                best_seg_metric = seg_val_epoch_results[seg_val_monitor_metric][-1]
                torch.save(unet.state_dict(), os.path.join(path_current, "fused_unet_best_alone.pth"))
                log.info("Saved a new best-metric unet model (alone)")
            if config['clf_update'] in ['both', 'ukb_only']:

                if clf_val_epoch_metrics[clf_val_monitor_metric][-1] > best_clf_metric:
                    best_clf_metric = clf_val_epoch_metrics[clf_val_monitor_metric][-1]
                    torch.save(clfNet.state_dict(), os.path.join(path_current, "clf_best_alone.pth"))
                    log.info("Saved a new best-metric clf model (alone)")
                avg_val_results.append(avg_val_monitor_metric(seg_val_epoch_results[seg_val_monitor_metric][-1],
                                                              clf_val_epoch_metrics[clf_val_monitor_metric][-1]))

                if avg_val_results[-1] > best_avg_metric:
                    best_avg_metric = avg_val_results[-1]
                    torch.save(unet.state_dict(), os.path.join(path_current, "unet_best_common.pth"))
                    torch.save(clfNet.state_dict(), os.path.join(path_current, "clf_best_common.pth"))
                    log.info("Saved a pair of new best-metric unet and clf model")

            seg_metric_val.reset()
            clf_metric_val.reset()
            fpr_metric_val.reset()
            del val_epoch_clf_loss, val_epoch_seg_loss, val_epoch_loss

    # save final model
    torch.save(unet.state_dict(), os.path.join(path_current, "unet_final_checkpoint.pth"))
    torch.save(clfNet.state_dict(), os.path.join(path_current, "clf_final_checkpoint.pth"))

    with open(os.path.join(path_current, "training_history.pkl"), "wb") as f:
        history = dict(seg_train=seg_train_epoch_results, seg_val=seg_val_epoch_results,
                       clf_train=clf_train_epoch_metrics, clf_val=clf_val_epoch_metrics,
                       fpr_vall=fpr_val_epoch_results)
        pickle.dump(history, f)
        f.close()


def train(config):
    """
    The complete training procedure.
    """
    path_param = config["save_path"]
    path_run = utils.create_run_folder(path_param)
    dataPreparer = HALOSDataPreparer(param=config)

    # record case ids that ware used for training and validation
    with open(os.path.join(path_run, "ids_train_val.pkl"), "wb") as f:
        data = dict(seg_train=dataPreparer.dict_seg_train, seg_val=dataPreparer.dict_seg_val,
                    ukb_train=dataPreparer.dict_ukb_train, ukb_val=dataPreparer.dict_ukb_val,
                    seg_train_formed=dataPreparer.dict_seg_train_formed,
                    seg_val_formed=dataPreparer.dict_seg_val_formed,
                    ukb_train_formed=dataPreparer.dict_ukb_train_formed,
                    ukb_val_formed=dataPreparer.dict_ukb_val_formed)
        pickle.dump(data, f)
        f.close()
    with open(os.path.join(path_run, "config.pkl"), "wb") as f:
        pickle.dump(config, f)
        f.close()

    if config["cross_validation"]:
        for i in range(config["cv_fold"]):
            train_fold(config, cv_index=i, path_run=path_run, dataPreparer=dataPreparer)
    else:
        train_fold(config, cv_index=0, path_run=path_run, dataPreparer=dataPreparer)


if __name__ == "__main__":
    config_default = HALOS_CONF.PARAM

    config = parse_config_halos(config_default)

    torch.manual_seed(config["random_seed"])
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])

    train(config)
