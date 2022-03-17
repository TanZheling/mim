import copy
import warnings

from mmcls.models import CLASSIFIERS, build_backbone, build_head, build_neck
from mmcls.models.utils import Augments
from mmcls.models.classifiers import BaseClassifier, ImageClassifier


@CLASSIFIERS.register_module()
class imageClassifier(ImageClassifier):

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                shoulf be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)

        self.feat = self.extract_feat(img)

        losses = dict()
        loss = self.head.forward_train(self.feat, gt_label)
        losses.update(loss)

        return losses

    def simple_test(self, img, **kwargs):
        """Test without augmentation."""
        self.feat = self.extract_feat(img)
        #self.feat_dims = len(self.feat.shape)
        #if self.feat_dims == 1:
        #    self.feat.unsqueeze_(0)
        return self.head.simple_test(self.feat, **kwargs)