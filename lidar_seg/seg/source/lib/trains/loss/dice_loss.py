import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import warnings
from torch.nn.modules.loss import _WeightedLoss

# We use these functions in torch/legacy as well, in which case we'll silence the warning
def legacy_get_string(size_average: Optional[bool], reduce: Optional[bool], emit_warning: bool = True) -> str:
    warning = "size_average and reduce args will be deprecated, please use reduction='{}' instead."

    if size_average is None:
        size_average = True
    if reduce is None:
        reduce = True

    if size_average and reduce:
        ret = 'mean'
    elif reduce:
        ret = 'sum'
    else:
        ret = 'none'
    if emit_warning:
        warnings.warn(warning.format(ret))
    return ret

class DiceLoss(_WeightedLoss):
    """
    This criterion is based on Dice coefficients.

    Modified version of: https://github.com/ai-med/nn-common-modules/blob/master/nn_common_modules/losses.py (MIT)
    Arxiv paper: https://arxiv.org/pdf/1606.04797.pdf
    """

    def __init__(
        self, 
        weight: Optional[torch.FloatTensor] = None,
        ignore_index: int = 255,
        binary: bool = False,
        reduction: str = 'mean'):
        """
        :param weight:  <torch.FloatTensor: n_class>. Optional scalar weight for each class.
        :param ignore_index: Label id to ignore when calculating loss.
        :param binary: Whether we are only doing binary segmentation.
        :param reduction: Specifies the reduction to apply to the output. Can be 'none', 'mean' or 'sum'.
        """
        super().__init__(weight=weight, reduction=reduction)
        self.ignore_index = ignore_index
        self.binary = binary

    def forward(self, predictions: torch.FloatTensor, targets: torch.LongTensor) -> torch.FloatTensor:
        """
        Forward pass.
        :param predictions: <torch.FloatTensor: n_samples, C, H, W>. Predicted scores.
        :param targets: <torch.LongTensor: n_samples, H, W>. Target labels.
        :return: <torch.FloatTensor: 1>. Scalar loss output.
        """
        self._check_dimensions(predictions=predictions, targets=targets)
        predictions = F.softmax(predictions, dim=1)
        if self.binary:
            return self._dice_loss_binary(predictions, targets)
        return self._dice_loss_multichannel(predictions, targets, self.weight, self.ignore_index)

    @staticmethod
    def _dice_loss_binary(predictions: torch.FloatTensor, targets: torch.LongTensor) -> torch.FloatTensor:
        """
        Dice loss for one channel binarized input.
        :param predictions: <torch.FloatTensor: n_samples, 1, H, W>. Predicted scores.
        :param targets: <torch.LongTensor: n_samples, H, W>. Target labels.
        :return: <torch.FloatTensor: 1>. Scalar loss output.
        """
        eps = 0.0001

        assert predictions.size(1) == 1, 'predictions should have a class size of 1 when doing binary dice loss.'

        intersection = predictions * targets

        # Summed over batch, height and width.
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = predictions + targets
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = 1 - (numerator / denominator)

        # Averaged by classes.
        return loss_per_channel.sum() / predictions.size(1)

    @staticmethod
    def _dice_loss_multichannel(predictions: torch.FloatTensor,
                                targets: torch.LongTensor,
                                weight: Optional[torch.FloatTensor] = None,
                                ignore_index: int = -100) -> torch.FloatTensor:
        """
        Calculate the loss for multichannel predictions.
        :param predictions: <torch.FloatTensor: n_samples, n_class, H, W>. Predicted scores.
        :param targets: <torch.LongTensor: n_samples, H, W>. Target labels.
        :param weight:  <torch.FloatTensor: n_class>. Optional scalar weight for each class.
        :param ignore_index: Label id to ignore when calculating loss.
        :return: <torch.FloatTensor: 1>. Scalar loss output.
        """
        eps = 0.0001
        encoded_target = predictions.detach() * 0

        mask = targets == ignore_index
        targets = targets.clone()
        targets[mask] = 0
        encoded_target.scatter_(1, targets.unsqueeze(1), 1)
        mask = mask.unsqueeze(1).expand_as(encoded_target)
        encoded_target[mask] = 0

        intersection = predictions * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = predictions + encoded_target

        denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1)
        if denominator.sum() == 0:
            # Means only void gradients. Summing the denominator would lead to loss of 0.
            return denominator.sum()
        denominator = denominator + eps

        if weight is None:
            weight = 1
        else:
            # We need to ensure that the weights and the loss terms resides in the same device id.
            # Especially crucial when we are using DataParallel/DistributedDataParallel.
            weight = weight / weight.mean()

        loss_per_channel = weight * (1 - (numerator / denominator))

        # Averaged by classes.
        return loss_per_channel.sum() / predictions.size(1)

    def _check_dimensions(self, predictions: torch.FloatTensor, targets: torch.LongTensor) -> None:
        error_message = ""
        if predictions.size(0) != targets.size(0):
            error_message += f'Predictions and targets should have the same batch size, but predictions have batch ' f'size {predictions.size(0)} and targets have batch size {targets.size(0)}.\n'
        if self.weight is not None and self.weight.size(0) != predictions.size(1):
            error_message += f'Weights and the second dimension of predictions should have the same dimensions ' f'equal to the number of classes, but weights has dimension {self.weight.size()} and ' f'targets has dimension {targets.size()}.\n'
        if self.binary and predictions.size(1) != 1:
            error_message += f'Binary class should have one channel representing the number of classes along the ' f'second dimension of the predictions, but the actual dimensions of the predictions ' f'is {predictions.size()}\n'
        if not self.binary and predictions.size(1) == 1:
            error_message += f'Predictions has dimension {predictions.size()}. The 2nd dimension equal to 1 ' f'indicates that this is binary, but binary was set to {self.binary} by construction\n'
        if error_message:
            raise ValueError(error_message)
