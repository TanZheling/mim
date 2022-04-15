import copy
import warnings

from mmcls.models import CLASSIFIERS, build_backbone, build_head, build_neck
from mmcls.models.utils import Augments
from mmcls.models.classifiers import BaseClassifier, ImageClassifier
import torch

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

    '''def extract_feat(self, img):
        """Directly extract features from the backbone + neck."""
        x = self.backbone(img)
        
        if isinstance(x, tuple):
            self.medium_level = x
            x = x[-1]
            
        else:
            self.medium_level = (x,)
        if self.with_neck:
            x = self.neck(x)
        return x'''

    def simple_test(self, img, img_metas=None, **kwargs):
        """Test without augmentation."""
        self.feat = self.extract_feat(img)
        if isinstance(self.feat, tuple):
            self.feat = self.feat[0]
        #self.feat_dims = len(self.feat.shape)
        #if self.feat_dims == 1:
            #self.feat.unsqueeze_(0)
        return self.head.simple_test(self.feat, **kwargs)