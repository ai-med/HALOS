from typing import Optional, List, Union

import torch
import torch.nn as nn
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.metrics.utils import do_metric_reduction
from monai.transforms import AsDiscrete
from sklearn.metrics import confusion_matrix


class CLFMetric:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.pred = []
        self.gt = []
        self.softmax = nn.Softmax(dim=1)
        if self.num_classes == 2:
            self.calculate_metrics = self.calculate_metrics_binary
            self.metrics = dict(
                Accuracy=[],
                Precision=[],
                Recall=[],
                F1=[],
                Balanced_Accuracy=[]
            )
        elif self.num_classes == 3:
            self.calculate_metrics = self.calculate_metrics_tri
            self.metrics = dict(
                Accuracy_LK=[],
                Precision_LK=[],
                Recall_LK=[],
                F1_LK=[],
                Balanced_Accuracy_LK=[],
                Accuracy_RK=[],
                Precision_RK=[],
                Recall_RK=[],
                F1_RK=[],
                Balanced_Accuracy_RK=[],
                Balanced_Accuracy=[]
            )
        else:
            print("Incorrect num_classes param!")

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        pred = self.softmax(pred)
        self.pred.append(pred.cpu())
        self.gt.append(gt.cpu())

    def reset(self):
        self.pred = []
        self.gt = []

    def compute_confusion_matrix_binary(self):
        pred = torch.cat(self.pred, 0)
        gt = torch.cat(self.gt, 0)
        cmat = confusion_matrix(y_true=gt, y_pred=pred.argmax(dim=1), labels=[0, 1])
        tn, fp, fn, tp = cmat.ravel()
        return tn, fp, fn, tp, cmat

    def compute_confusion_matrix_tri(self, positive_class_index):
        pred = torch.cat(self.pred, 0)
        pred_argmax = pred.argmax(dim=1)
        pred_binary = [1 if i == positive_class_index else 0 for i in pred_argmax]
        gt = torch.cat(self.gt, 0)
        gt_binary = [1 if i == positive_class_index else 0 for i in gt]
        cmat = confusion_matrix(y_true=gt_binary, y_pred=pred_binary, labels=[0, 1])
        tn, fp, fn, tp = cmat.ravel()
        return tn, fp, fn, tp, cmat

    @staticmethod
    def _accuracy(tn, fp, fn, tp):
        return (tp + tn) / (tp + tn + fp + fn)

    @staticmethod
    def _precision(fp, tp):
        return tp / (tp + fp)

    @staticmethod
    def _recall(fn, tp):
        return tp / (tp + fn)

    @staticmethod
    def _f1_score(precision, recall):
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def _balanced_accuracy(cmat):
        cmat = torch.Tensor(cmat)
        per_class = cmat.diag() / cmat.sum(dim=1)
        per_class = per_class[~torch.isnan(per_class)]
        bacc = per_class.mean()
        return bacc

    def calculate_metrics_binary(self):
        tn, fp, fn, tp, cmat = self.compute_confusion_matrix_binary()
        precision = self._precision(fp, tp)
        recall = self._recall(fn, tp)
        self.metrics["Accuracy"].append(self._accuracy(tn, fp, fn, tp))
        self.metrics["Precision"].append(precision)
        self.metrics["Recall"].append(recall)
        self.metrics["F1"].append(self._f1_score(precision, recall))
        self.metrics["Balanced_Accuracy"].append(self._balanced_accuracy(cmat))
        return self.metrics

    def calculate_metrics_tri(self):
        tn_lk, fp_lk, fn_lk, tp_lk, cmat_lk = self.compute_confusion_matrix_tri(positive_class_index=1)
        precision_lk = self._precision(fp_lk, tp_lk)
        recall_lk = self._recall(fn_lk, tp_lk)
        bacc_lk = self._balanced_accuracy(cmat_lk)
        self.metrics["Accuracy_LK"].append(self._accuracy(tn_lk, fp_lk, fn_lk, tp_lk))
        self.metrics["Precision_LK"].append(precision_lk)
        self.metrics["Recall_LK"].append(recall_lk)
        self.metrics["F1_LK"].append(self._f1_score(precision_lk, recall_lk))
        self.metrics["Balanced_Accuracy_LK"].append(bacc_lk)

        tn_rk, fp_rk, fn_rk, tp_rk, cmat_rk = self.compute_confusion_matrix_tri(positive_class_index=2)
        precision_rk = self._precision(fp_rk, tp_rk)
        recall_rk = self._recall(fn_rk, tp_rk)
        bacc_rk = self._balanced_accuracy(cmat_rk)
        self.metrics["Accuracy_RK"].append(self._accuracy(tn_rk, fp_rk, fn_rk, tp_rk))
        self.metrics["Precision_RK"].append(precision_rk)
        self.metrics["Recall_RK"].append(recall_rk)
        self.metrics["F1_RK"].append(self._f1_score(precision_rk, recall_rk))
        self.metrics["Balanced_Accuracy_RK"].append(bacc_rk)

        avg_bacc = (bacc_lk + bacc_rk) / 2
        self.metrics["Balanced_Accuracy"].append(avg_bacc)
        return self.metrics


class SegMetric:
    def __init__(self, organ="gallbladder", mode="train", save_individual_result=False):
        self.organ = organ
        self.mode = mode
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", ignore_empty=False)
        self.dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch", ignore_empty=False)
        self.FPR_metric = FalsePositiveRate(organ=self.organ)
        self.results = {"average": [], "liver": [], "spleen": [], "right_kidney": [], "left_kidney": [],
                        "pancreas": [], "gallbladder": [], "FPR_seg": []}
        self.save_individual_result = save_individual_result
        if self.save_individual_result:
            self.results_individual = {}
        self.post_transform_label = AsDiscrete(to_onehot=7)

    def _post_transform(self, label):
        labels_list = decollate_batch(label)
        labels_convert = [self.post_transform_label(i) for i in labels_list]

        return labels_convert

    def reset(self):
        self.dice_metric.reset()
        self.dice_metric_batch.reset()
        self.FPR_metric.reset()

    def compute_metrics(self, preds: Union[torch.Tensor, List[torch.Tensor]],
                        label: Union[torch.Tensor, List[torch.Tensor]], binary_label: torch.Tensor,
                        img_id: Optional[str]
                        ):
        if self.mode == "train":

            B = preds[0].shape[0]
            post_transform = AsDiscrete(argmax=True, to_onehot=7)
            labels_convert = [label[0][i] for i in range(B)]
            preds_convert = [post_transform(preds[0][i]) for i in range(B)]
        else:
            labels_convert = self._post_transform(label)
            preds_convert = preds
        self.dice_metric_batch(y_pred=preds_convert, y=labels_convert)
        self.dice_metric(y_pred=preds_convert, y=labels_convert)
        # debugging
        dice = self.dice_metric.get_buffer()
        print('dice ', dice)
        self.FPR_metric.compute_metric(pred=preds_convert, binary_label=binary_label)
        if self.save_individual_result:
            self.results_individual[img_id] = {}
            results_individual = self.dice_metric_batch._buffers[0][-1].tolist()
            if self.organ == "gallbladder":
                if binary_label == 1:
                    if preds_convert[0][-1].sum() >= 1:
                        self.results_individual[img_id]["FP"] = True
                    else:
                        self.results_individual[img_id]["FP"] = False
                else:
                    self.results_individual[img_id]["FP"] = None
            elif self.organ == "kidney":
                if binary_label == 1:
                    if preds_convert[0][4].sum() >= 1:
                        self.results_individual[img_id]["FP"] = True
                    else:
                        self.results_individual[img_id]["FP"] = False
                elif binary_label == 2:
                    if preds_convert[0][3].sum() >= 1:
                        self.results_individual[img_id]["FP"] = True
                    else:
                        self.results_individual[img_id]["FP"] = False
                else:
                    self.results_individual[img_id]["FP"] = None
            else:
                print("Spleen UKB dataset not yet created!!")
            self.results_individual[img_id]["liver"] = results_individual[0]
            self.results_individual[img_id]["spleen"] = results_individual[1]
            self.results_individual[img_id]["right_kidney"] = results_individual[2]
            self.results_individual[img_id]["left_kidney"] = results_individual[3]
            self.results_individual[img_id]["pancreas"] = results_individual[4]
            self.results_individual[img_id]["gallbladder"] = results_individual[5]

    def update_metrics(self):
        print('average_dice  ', self.dice_metric.get_buffer())
        print('average_dice  ', self.dice_metric.get_buffer().dtype)
        average_dice = self.dice_metric.aggregate()
        print('average_dice ', average_dice)
        classwise_dice = self.dice_metric_batch.aggregate()
        print('classwise ', classwise_dice)

        new_cw_dice = do_metric_reduction(self.dice_metric.get_buffer(), 'mean')
        print('new cw dice ', new_cw_dice)
        fpr = self.FPR_metric.update_metric()
        self.results["average"].append(average_dice)
        self.results["liver"].append(classwise_dice[0])
        self.results["spleen"].append(classwise_dice[1])
        self.results["right_kidney"].append(classwise_dice[2])
        self.results["left_kidney"].append(classwise_dice[3])
        self.results["pancreas"].append(classwise_dice[4])
        self.results["gallbladder"].append(classwise_dice[5])
        self.results["FPR_seg"].append(fpr)


class FalsePositiveRate:
    def __init__(self, organ="gallbladder"):
        self.FP = 0
        self.TN = 0
        self.result = 0
        self.organ = organ
        if self.organ == "gallbladder":
            self.compute_metric = self.compute_metric_gallbladder
        elif self.organ == "kidney":
            self.compute_metric = self.compute_metric_kidney
        else:
            print("Spleen UKB dataset not yet created!!")

    def reset(self):
        self.FP = 0
        self.TN = 0
        self.result = 0

    def compute_metric_gallbladder(self, pred, binary_label):
        for b in range(len(binary_label)):
            if binary_label[b] == 1:
                if pred[b][-1].sum() >= 1:
                    fp, tn = 1, 0
                else:
                    fp, tn = 0, 1
            else:
                fp, tn = 0, 0
            self.FP += fp
            self.TN += tn

    def compute_metric_kidney(self, pred, binary_label):
        for b in range(len(binary_label)):
            if binary_label[b] == 1:
                if pred[b][4].sum() >= 1:
                    fp, tn = 1, 0
                else:
                    fp, tn = 0, 1
            elif binary_label[b] == 2:
                if pred[b][3].sum() >= 1:
                    fp, tn = 1, 0
                else:
                    fp, tn = 0, 1
            else:
                fp, tn = 0, 0
            self.FP += fp
            self.TN += tn

    def update_metric(self):
        self.result = self.FP / (self.FP + self.TN + 0.000000001)
        return self.result


class GallbladderOnly:
    def __init__(self, save_individual_result=False):
        self.TP = 0
        self.FN = 0
        self.FP = 0
        self.TN = 0
        self.FPR = 0
        self.results = {"TP": [], "FN": [], "FP": [], "TN": [], "FPR_ukb": []}
        self.save_individual_result = save_individual_result
        if self.save_individual_result:
            self.results_individual = {}

    def reset(self):
        self.TP = 0
        self.FN = 0
        self.FP = 0
        self.TN = 0
        self.FPR = 0

    def compute_metrics(self, preds_convert: Union[torch.Tensor, List[torch.Tensor]], binary_label: torch.Tensor,
                        img_id: Optional[str]):

        B = preds_convert[0].shape[0]
        post_transform = AsDiscrete(argmax=True, to_onehot=7)
        binary_label = [binary_label[i] for i in range(B)]
        preds_convert = [post_transform(preds_convert[0][i]) for i in range(B)]
        # else:
        #    labels_convert = self._post_transform(binary_label)
        #    #preds_convert = preds_convert

        self._compute_metric(pred=preds_convert, binary_label=binary_label)
        if self.save_individual_result:
            self.results_individual[img_id] = {}
            if binary_label == 1:
                if preds_convert[0][-1].sum() >= 1:
                    self.results_individual[img_id]["FP"] = torch.sum(preds_convert[0][-1])
                else:
                    self.results_individual[img_id]["FP"] = False
            else:
                self.results_individual[img_id]["FP"] = None

    def update_metrics(self):
        fpr = self._update_fpr_metric()
        self.results["TP"].append(self.TP)
        self.results["FN"].append(self.FN)
        self.results["FP"].append(self.FP)
        self.results["TN"].append(self.TN)
        self.results["FPR_ukb"].append(fpr)

    def _compute_metric(self, pred, binary_label):
        for b in range(len(binary_label)):
            if binary_label[b] == 1:
                if pred[b][-1].sum() >= 1:
                    fp, tn = 1, 0
                else:
                    fp, tn = 0, 1
                tp, fn = 0, 0
            else:
                if pred[b][-1].sum() >= 1:
                    tp, fn = 1, 0
                else:
                    tp, fn = 0, 1
                fp, tn = 0, 0
            self.FP += fp
            self.TN += tn
            self.TP += tp
            self.FN += fn

    def _update_fpr_metric(self):
        self.FPR = self.FP / (self.FP + self.TN)
        return self.FPR


class KidneyOnly:
    def __init__(self, save_individual_result=False):
        self.TP = 0
        self.FN = 0
        self.FP = 0
        self.TN = 0
        self.FPR = 0
        self.post_transform = AsDiscrete(argmax=True, to_onehot=7)
        self.results = {"TP": [], "FN": [], "FP": [], "TN": [], "FPR_ukb": []}
        self.save_individual_result = save_individual_result
        if self.save_individual_result:
            self.results_individual = {}

    def reset(self):
        self.TP = 0
        self.FN = 0
        self.FP = 0
        self.TN = 0
        self.FPR = 0

    def compute_metrics(self, preds_convert: Union[torch.Tensor, List[torch.Tensor]], binary_label: torch.Tensor,
                        img_id: Optional[str]):
        B = preds_convert[0].shape[0]
        preds_convert = [self.post_transform(preds_convert[0][i]) for i in range(B)]

        self._compute_metric(pred=preds_convert, binary_label=binary_label)
        if self.save_individual_result:
            self.results_individual[img_id] = {}
            if binary_label == 1:
                if preds_convert[0][4].sum() >= 1:
                    self.results_individual[img_id]["FP"] = torch.sum(preds_convert[0][-1])
                else:
                    self.results_individual[img_id]["FP"] = False
            elif binary_label == 2:
                if preds_convert[0][3].sum() >= 1:
                    self.results_individual[img_id]["FP"] = torch.sum(preds_convert[0][-1])
                else:
                    self.results_individual[img_id]["FP"] = False
            else:
                self.results_individual[img_id]["FP"] = None

    def compute_metrics_eval(self, preds_convert: Union[torch.Tensor, List[torch.Tensor]], binary_label: torch.Tensor,
                             img_id: Optional[str]):
        preds_convert = [preds_convert[i] for i in range(len(preds_convert))]

        self._compute_metric(pred=preds_convert, binary_label=binary_label)
        if self.save_individual_result:
            self.results_individual[img_id] = {}
            if binary_label == 1:
                if preds_convert[0][4].sum() >= 1:
                    self.results_individual[img_id]["FP"] = torch.sum(preds_convert[0][-1])
                else:
                    self.results_individual[img_id]["FP"] = False
            elif binary_label == 2:
                if preds_convert[0][3].sum() >= 1:
                    self.results_individual[img_id]["FP"] = torch.sum(preds_convert[0][-1])
                else:
                    self.results_individual[img_id]["FP"] = False
            else:
                self.results_individual[img_id]["FP"] = None

    def update_metrics(self):
        fpr = self._update_fpr_metric()
        self.results["TP"].append(self.TP)
        self.results["FN"].append(self.FN)
        self.results["FP"].append(self.FP)
        self.results["TN"].append(self.TN)
        self.results["FPR_ukb"].append(fpr)

    def _compute_metric(self, pred, binary_label):
        for b in range(len(binary_label)):
            if binary_label[b] == 1:
                if pred[b][4].sum() >= 1:
                    fp, tn = 1, 0
                else:
                    fp, tn = 0, 1
                tp, fn = 0, 0
            elif binary_label[b] == 2:
                if pred[b][3].sum() >= 1:
                    fp, tn = 1, 0
                else:
                    fp, tn = 0, 1
                tp, fn = 0, 0
            else:
                if pred[b][4].sum() >= 1 and pred[b][3].sum() >= 1:
                    tp, fn = 1, 0
                else:
                    tp, fn = 0, 1
                fp, tn = 0, 0
            self.FP += fp
            self.TN += tn
            self.TP += tp
            self.FN += fn

    def _update_fpr_metric(self):
        self.FPR = self.FP / (self.FP + self.TN + 0.000000001)
        return self.FPR


class KidneyOnly_separate:
    def __init__(self, save_individual_result=False):
        self.TP_lk = 0
        self.FN_lk = 0
        self.FP_lk = 0
        self.TN_lk = 0
        self.FPR_lk = 0
        self.TP_rk = 0
        self.FN_rk = 0
        self.FP_rk = 0
        self.TN_rk = 0
        self.FPR_rk = 0
        self.post_transform = AsDiscrete(argmax=True, to_onehot=7)
        self.results = {"TP_lk": [], "FN_lk": [], "FP_lk": [], "TN_lk": [], "FPR_lk_ukb": [],
                        "TP_rk": [], "FN_rk": [], "FP_rk": [], "TN_rk": [], "FPR_rk_ukb": []}
        self.save_individual_result = save_individual_result
        if self.save_individual_result:
            self.results_individual = {}

    def reset(self):
        self.TP_lk = 0
        self.FN_lk = 0
        self.FP_lk = 0
        self.TN_lk = 0
        self.FPR_lk = 0
        self.TP_rk = 0
        self.FN_rk = 0
        self.FP_rk = 0
        self.TN_rk = 0
        self.FPR_rk = 0

    def compute_metrics(self, preds_convert: Union[torch.Tensor, List[torch.Tensor]], binary_label: torch.Tensor,
                        img_id: Optional[str]):
        B = preds_convert[0].shape[0]
        preds_convert = [self.post_transform(preds_convert[0][i]) for i in range(B)]

        self._compute_metric(pred=preds_convert, binary_label=binary_label)
        if self.save_individual_result:
            self.results_individual[img_id] = {}
            if binary_label == 1:
                if preds_convert[0][4].sum() >= 1:
                    self.results_individual[img_id]["FP"] = torch.sum(preds_convert[0][-1])
                else:
                    self.results_individual[img_id]["FP"] = False
            elif binary_label == 2:
                if preds_convert[0][3].sum() >= 1:
                    self.results_individual[img_id]["FP"] = torch.sum(preds_convert[0][-1])
                else:
                    self.results_individual[img_id]["FP"] = False
            else:
                self.results_individual[img_id]["FP"] = None

    def compute_metrics_eval(self, preds_convert: Union[torch.Tensor, List[torch.Tensor]], binary_label: torch.Tensor,
                             img_id: Optional[str]):
        preds_convert = [preds_convert[i] for i in range(len(preds_convert))]

        self._compute_metric(pred=preds_convert, binary_label=binary_label)
        if self.save_individual_result:
            self.results_individual[img_id] = {}
            if binary_label == 1:
                if preds_convert[0][4].sum() >= 1:
                    self.results_individual[img_id]["FP_lk"] = torch.sum(preds_convert[0][-1])
                else:
                    self.results_individual[img_id]["FP"] = False
            elif binary_label == 2:
                if preds_convert[0][3].sum() >= 1:
                    self.results_individual[img_id]["FP_rk"] = torch.sum(preds_convert[0][-1])
                else:
                    self.results_individual[img_id]["FP"] = False
            else:
                self.results_individual[img_id]["FP"] = None

    def update_metrics(self):
        fpr_lk, fpr_rk = self._update_fpr_metric()
        self.results["TP_lk"].append(self.TP_lk)
        self.results["FN_lk"].append(self.FN_lk)
        self.results["FP_lk"].append(self.FP_lk)
        self.results["TN_lk"].append(self.TN_lk)
        self.results["FPR_lk_ukb"].append(fpr_lk)

        self.results["TP_rk"].append(self.TP_rk)
        self.results["FN_rk"].append(self.FN_rk)
        self.results["FP_rk"].append(self.FP_rk)
        self.results["TN_rk"].append(self.TN_rk)
        self.results["FPR_rk_ukb"].append(fpr_rk)

    def _compute_metric(self, pred, binary_label):
        for b in range(len(binary_label)):
            if binary_label[b] == 1:
                if pred[b][4].sum() >= 1:
                    fp_lk, tn_lk = 1, 0
                else:
                    fp_lk, tn_lk = 0, 1
                tp_lk, fn_lk = 0, 0
                tp_rk, fn_rk = 0, 0
                fp_rk, tn_rk = 0, 0
            elif binary_label[b] == 2:
                if pred[b][3].sum() >= 1:
                    fp_rk, tn_rk = 1, 0
                else:
                    fp_rk, tn_rk = 0, 1
                tp_rk, fn_rk = 0, 0
                tp_lk, fn_lk = 0, 0
                fp_lk, tn_lk = 0, 0
            else:
                if pred[b][4].sum() >= 1 and pred[b][3].sum() >= 1:
                    tp_lk, fn_lk = 1, 0
                    tp_rk, fn_rk = 1, 0
                elif pred[b][4].sum() < 1 and pred[b][3].sum() >= 1:
                    tp_lk, fn_lk = 0, 1
                    tp_rk, fn_rk = 1, 0
                elif pred[b][3].sum() < 1 and pred[b][4].sum() >= 1:
                    tp_rk, fn_rk = 0, 1
                    tp_lk, fn_lk = 1, 0
                else:
                    tp_lk, fn_lk = 0, 1
                    tp_rk, fn_rk = 0, 1
                fp_lk, tn_lk = 0, 0
                fp_rk, tn_rk = 0, 0
            self.FP_lk += fp_lk
            self.TN_lk += tn_lk
            self.TP_lk += tp_lk
            self.FN_lk += fn_lk
            self.FP_rk += fp_rk
            self.TN_rk += tn_rk
            self.TP_rk += tp_rk
            self.FN_rk += fn_rk

    def _update_fpr_metric(self):
        self.FPR_lk = self.FP_lk / (self.FP_lk + self.TN_lk + 0.000000001)
        self.FPR_rk = self.FP_rk / (self.FP_rk + self.TN_rk + 0.000000001)
        return self.FPR_lk, self.FPR_rk
