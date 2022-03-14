import torch.nn.functional as F
from mmcls.models.heads import LinearClsHead
from mmcls.models.builder import HEADS

@HEADS.register_module()
class linearClsHead(LinearClsHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 requires_grad=True,
                 *args,
                 **kwargs):
        super(linearClsHead, self).__init__(
            num_classes,
            in_channels,
            *args,
            **kwargs)

        if not requires_grad:
            for param in self.fc.parameters():
                param.requires_grad = False
    
    def forward_train(self, x, gt_label):
        if isinstance(x, tuple):
            x = x[-1]
        self.cls_score = self.fc(x)
        losses = self.loss(self.cls_score, gt_label)
        return losses

    def simple_test(self, img, without_softmax=False, **kwargs):
        """Test without augmentation."""
        cls_score = self.fc(img)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        if without_softmax:
            return cls_score
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None

        return self.post_process(pred)