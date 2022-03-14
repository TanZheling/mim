import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
#from mmcv.ops import softmax_entropy
from mmcls.models.builder import LOSSES
from mmcls.models.losses import weight_reduce_loss

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@LOSSES.register_module()
class SoftmaxEntropyLoss(nn.Module):
    """Softmax entropy loss
    Args:
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, *args, **kwargs):
        super(SoftmaxEntropyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.cls_criterion = softmax_entropy

    def forward(self,
                cls_score,
                gt_label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None:
            weight = weight.float()
        loss_cls = self.loss_weight * weight_reduce_loss(
                    self.cls_criterion(cls_score),
                    weight=weight,
                    reduction=reduction,
                    avg_factor=avg_factor)
        return loss_cls