import argparse


def parse_config_halos(default):
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--path_seg_data", type=str, default=default["path_seg_data"])
    parser.add_argument("--path_ukb_data", type=str, default=default["path_ukb_data"])
    parser.add_argument("--path_ukb_label", type=str, default=default["path_ukb_label"])
    parser.add_argument("--organ", type=str, default=default["organ"], choices=["gallbladder", "kidney"])
    parser.add_argument("--key", type=str, default=default["key"])
    parser.add_argument("--clf_ukb_weight", nargs="+", type=float, default=default["clf_ukb_weight"])
    parser.add_argument("--random_seed", type=int, default=default["random_seed"])
    parser.add_argument("--device", type=str, default=default["device"])
    parser.add_argument("--max_epochs", type=int, default=default["max_epochs"])
    parser.add_argument("--val_interval", type=int, default=default["val_interval"])
    # unet
    parser.add_argument("--model", type=str, default=default["model"], choices=["nnUNet"])
    parser.add_argument("--enc_roi_size", nargs="+", type=int, default=default["enc_roi_size"])
    parser.add_argument("--val_number_pos_neg", nargs="+", type=int, default=default["val_number_pos_neg"])
    parser.add_argument("--seg_val_monitor_metric", type=str, default=default["seg_val_monitor_metric"],
                        choices=["average", "liver", "spleen", "right_kidney", "left_kidney", "pancreas", "gallbladder",
                                 "FPR"])
    parser.add_argument("--base_num_features", type=int, default=default["base_num_features"], choices=[8, 16, 32])

    # clf
    parser.add_argument("--clf_feature_loca", type=int, default=default["clf_feature_loca"],
                        choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--clf_model", type=str, default=default["clf_model"],
                        choices=["CLFDense", "CLFCNN", "CLFMixed", 'simple'])
    parser.add_argument("--clf_dropout", type=float, default=default["clf_dropout"])
    parser.add_argument("--clf_channels_conv", nargs="+", type=int, default=default["clf_channels_conv"])
    parser.add_argument("--clf_channels_dense", nargs="+", type=int, default=default["clf_channels_dense"])
    parser.add_argument("--clf_train_val_ratio", nargs="+", type=int, default=default["clf_train_val_ratio"])
    parser.add_argument("--clf_val_monitor_metric", type=str, default=default["clf_val_monitor_metric"],
                        choices=["Accuracy", "Precision", "Recall", "F1", "Balanced_Accuracy"])
    # feature fusion
    parser.add_argument("--fusion_loca", type=int, default=default["fusion_loca"], choices=[1, 2, 3])
    parser.add_argument("--fusion_squeeze_factor", type=int, default=default["fusion_squeeze_factor"])
    # training
    parser.add_argument("--cross_validation", type=bool, default=default["cross_validation"])
    parser.add_argument("--cv_fold", type=int, default=default["cv_fold"])
    parser.add_argument("--loss_weight_seg", type=float, default=default["loss_weight_seg"])
    parser.add_argument("--seg_lr", type=float, default=default["seg_lr"])
    parser.add_argument("--clf_lr", type=float, default=default["clf_lr"])

    parser.add_argument("--optimizer", type=str, default=default["optimizer"], choices=["Adam", "SGD", "AdamW"])
    parser.add_argument("--weight_decay", type=float, default=default["weight_decay"])
    parser.add_argument("--num_classes", type=int, default=default["num_classes"])
    parser.add_argument("--clf_factor", type=int, default=default["clf_factor"])
    parser.add_argument("--batch_size", type=int, default=default["batch_size"])
    parser.add_argument("--val_batch_size", type=int, default=default["val_batch_size"])
    parser.add_argument("--lr", type=float, default=default["lr"])

    parser.add_argument("--clf_update", type=str, default=default["clf_update"], choices=["ukb_only", "both", 'none'])
    parser.add_argument("--cache_rate", type=float, default=default["cache_rate"])
    parser.add_argument("--fusion_input", type=str, default=default["fusion_input"], choices=["gt", "clf"])
    parser.add_argument("--save_path", type=str, default=default["save_path"])
    parser.add_argument("--norm", type=str, default=default["norm"], choices=["IN", "BN"])
    parser.add_argument("--dropout_p", type=float, default=default["dropout_p"])
    parser.add_argument("--dropout_in_localization", default=default["dropout_in_localization"], action="store_true")

    parse_config = parser.parse_args()
    return vars(parse_config)
