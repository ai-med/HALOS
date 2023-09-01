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
import warnings
from typing import Callable, Optional

import torch
import torch.nn as nn
from monai.losses import DiceLoss
from monai.utils import DiceCEReduction, look_up_option, pytorch_after
from torch.nn.modules.loss import _Loss


class WeightedDiceCELoss(_Loss):
    """
    Adapted from Monai DiceCELoss

    Compute both Dice loss and Cross Entropy Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of Cross Entropy Loss is shown in ``torch.nn.CrossEntropyLoss``. In this implementation,
    two deprecated parameters ``size_average`` and ``reduce``, and the parameter ``ignore_index`` are
    not supported.

    """

    def __init__(
            self,
            include_background: bool = True,
            to_onehot_y: bool = False,
            sigmoid: bool = False,
            softmax: bool = False,
            other_act: Optional[Callable] = None,
            squared_pred: bool = False,
            jaccard: bool = False,
            reduction: str = "mean",
            smooth_nr: float = 1e-5,
            smooth_dr: float = 1e-5,
            batch: bool = False,
            ce_weight: Optional[torch.Tensor] = None,
            lambda_dice: float = 1.0,
            lambda_ce: float = 1.0,
    ) -> None:
        """
        Args:
            ``ce_weight`` and ``lambda_ce`` are only used for cross entropy loss.
            ``reduction`` is used for both losses and other parameters are only used for dice loss.

            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss`.
            softmax: if True, apply a softmax function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss`.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example: `other_act = torch.tanh`.
                only used by the `DiceLoss`, don't need to specify activation function for `CrossEntropyLoss`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``. The dice loss should
                as least reduce the spatial dimensions, which is different from cross entropy loss, thus here
                the ``none`` option cannot be used.

                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            ce_weight: a rescaling weight given to each class for cross entropy loss.
                See ``torch.nn.CrossEntropyLoss()`` for more information.
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_ce: the trade-off weight value for cross entropy loss. The value should be no less than 0.0.
                Defaults to 1.0.

        """
        super().__init__()
        self.reduction = look_up_option(reduction, DiceCEReduction).value
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction='none',
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )
        self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight, reduction=self.reduction)
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_ce < 0.0:
            raise ValueError("lambda_ce should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.old_pt_ver = not pytorch_after(1, 10)
        self.batch = batch

    def ce(self, input: torch.Tensor, target: torch.Tensor):
        """
        Compute CrossEntropy loss for the input and target.
        Will remove the channel dim according to PyTorch CrossEntropyLoss:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?#torch.nn.CrossEntropyLoss.

        """
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch and n_target_ch == 1:
            target = torch.squeeze(target, dim=1)
            target = target.long()
        elif self.old_pt_ver:
            warnings.warn(
                f"Multichannel targets are not supported in this older Pytorch version {torch.__version__}. "
                "Using argmax (as a workaround) to convert target to a single channel."
            )
            target = torch.argmax(target, dim=1)
        elif not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)

        return self.cross_entropy(input, target)

    def forward(self, input: torch.Tensor, target: torch.Tensor, dice_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """

        if len(input.shape) != len(target.shape):
            raise ValueError("the number of dimensions for input and target should be the same.")
        # input shape: 2,7,160,160,96
        # target shape: 2,7,160,160,96
        # dice loss shape (batch =False): 2, 7, 1, 1, 1
        # dice loss shape (batch = True): 7, 1, 1, 1
        # dice weights shape: 2,7
        dice_loss = self.dice(input, target)
        # print('in diceCE loss ')

        # print('dice loss ', dice_loss.shape)
        # print(dice_loss)

        # print('weights ', dice_weights.shape)
        # print(dice_weights)

        # weight:
        if self.batch:
            # dice loss was computed for the whole batch, so we will take the mean of the weights over batch dim
            dice_weights = torch.mean(dice_weights, dim=0)
            dice_loss = dice_loss.squeeze(1).squeeze(1).squeeze(1) * dice_weights
        else:
            # dice loss was computed separately for each item in the batch
            dice_loss = dice_weights * dice_loss.squeeze(2).squeeze(2).squeeze(2)
        # dice reduction:
        if self.reduction == DiceCEReduction.MEAN.value:
            dice_loss = torch.mean(dice_loss)  # the batch and channel average
        elif self.reduction == DiceCEReduction.SUM.value:
            dice_loss = torch.sum(dice_loss)  # sum over the batch and channel dims
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum"].')
        ce_loss = self.ce(input, target)
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss
        # total_loss: torch.Tensor = self.lambda_dice * dice_loss

        return total_loss


class MultipleOutputLoss(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        Implementation based on nnUNet.
        Use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss: The explicit loss function.
        :param weight_factors: To enable deep supervision by computing the weighted sum of DiceCE losses at different
        scales.
        """
        super(MultipleOutputLoss, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y, loss_weights=None):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors
        if loss_weights is None:
            l = weights[0] * self.loss(x[0], y[0])
        else:
            l = weights[0] * self.loss(x[0], y[0], loss_weights)
        for i in range(1, len(x)):
            if weights[i] != 0:
                if loss_weights is None:
                    l += weights[i] * self.loss(x[i], y[i])
                else:
                    l += weights[i] * self.loss(x[i], y[i], loss_weights)
        return l
