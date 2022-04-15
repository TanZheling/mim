import torch
from torch import nn
import torch.nn.functional as F
import copy
import numpy as np

from mmcv.ops import softmax_entropy, info_max

from mmcv.runner import OptimizerHook, build_optimizer
from mmcv.parallel import MMDataParallel
from mmcv.runner import HOOKS
from mmcls.datasets.pipelines import Compose
from mmcls.models import build_classifier

# used as an optimizer hook
@HOOKS.register_module()
class tentOptimizerHook(OptimizerHook):
    def __init__(self,
                 optimizer_cfg=None,
                 loss_cfg=None,  # type in ['kl_div', 'infomax', 'entropy', 'PGC']
                 augment_cfg=None,
                 grad_clip=None,
                 reset=None,  # [None, 'batch', 'sample']
                 repeat=1):
        self.optimizer_cfg = optimizer_cfg
        self.loss_cfg = loss_cfg if loss_cfg!=None else dict(type='entropy')
        self.grad_clip = grad_clip
        self.reset = reset
        self.repeat = repeat
        self.mode = self.loss_cfg['mode']
        
        assert repeat > 0

        if 'entropy' in self.mode:
            self.entropy_weight = self.loss_cfg.get('entropy_weight', 1)
            self.entropy_type = self.loss_cfg.get('entropy_type', 'entropy')
            self.img_aug = self.loss_cfg.get('img_aug', 'weak')
            self.B = self.loss_cfg.get('B', 1)

        if 'contrast' in self.mode:
            self.model_cfg = self.loss_cfg.model_cfg
            self.origin = Compose(self.loss_cfg.origin_pipeline) if 'origin_pipeline' in self.loss_cfg else None
            self.contrast_weight = self.loss_cfg.get('contrast_weight', 1)
            self.projector_dim = self.loss_cfg.get('projector_dim', 10)
            self.class_num = self.loss_cfg.get('class_num', 100)
            self.queue_size = self.loss_cfg.get('queue_size', 1)
            self.norm = self.loss_cfg.get('norm', 'L1Norm')
            self.temp = self.loss_cfg.get('temp', 0.07)
            self.momentum = 0.999

            self.func = self.loss_cfg.get('func', 'best')
            self.CLUE = self.loss_cfg.get('CLUE', False)
            self.alpha = self.loss_cfg.get('alpha', 0.21)

        if 'cls' in self.mode:
            self.cls_weight = self.loss_cfg.get('cls_weight', 1)
            self.cls_type = self.loss_cfg.get('cls_type', 'weak')
            self.ce_type = self.loss_cfg.get('ce_type', 'smoothed')
            self.class_num = self.loss_cfg.get('class_num', 1000)

        if augment_cfg is not None:
            print("Generating the augmentation.")
            self.augment = Compose(augment_cfg)

    def before_run(self, runner):
        self.device = runner.model.device_ids[0]
        self.device_list = runner.model.device_ids
        if 'contrast' in self.mode:
            self.init_encoder(runner)
            self.init_bank(runner)
        if self.reset:
            runner.logger.info("Storing the related states.")
            self.model_state = copy.deepcopy(runner.model.state_dict())
            self.optimizer_state = copy.deepcopy(runner.optimizer.state_dict())

        self.device_init = True

    def before_train_epoch(self, runner):
        corr = runner.data_loader.dataset.corruption
        sev = runner.data_loader.dataset.severity
        if len(corr) > 1:
            self.acc_var = 'multi/online_accuracy_top-1'
            self.total_var = 'multi/online_total_num'
        else:
            self.acc_var = str(sev[0]) + '/online_accuracy_top-1'
            self.total_var = str(sev[0]) + '/online_total_num'
        self.num_pos, self.num_tot = 0, 0

    def before_train_iter(self, runner):
        '''if self.device_init:
            self.device = runner.data_batch['img'].device
            if 'contrast' in self.mode:
                self.init_encoder(runner)
                self.init_bank(runner)
            self.device_init = False'''

        if self.reset:
            runner.model.load_state_dict(self.model_state, strict=True)
            if 'contrast' in self.mode:
                self.encoder.load_state_dict(self.model_state, strict=True)
            runner.optimizer.load_state_dict(self.optimizer_state)

        if self.reset == 'sample':
            runner.model = self.configure_samplenorm(runner.data_batch['img'].size(0), runner.model)
            if 'contrast' in self.mode:
                self.encoder = self.configure_samplenorm(runner.data_batch['img'].size(0), self.encoder)
            runner.optimizer = build_optimizer(runner.model, self.optimizer_cfg)
        
    def after_train_iter(self, runner):
        '''
            test-time entropy optimization at the flow of 'train'
            variables:
                runner.model: MMDataParallel
                    module: ImageClassifier
                        head: LinearClsHead
                            cls_score: torch.Size([bs, num_classes]), e.g., [128, 10]
                        feat: torch.Size([bs, feature_dim]), e.g., [128, 2048]

                runner.outputs: dict
                    'loss': tensor, e.g., tensor(0.8785, device='cuda:7', grad_fn=<AddBackward0>)
                    'log_vars':OrderedDict,
                        'top-1': float, e.g., 79.6875
                        'loss':  float, e.g., 0.8784552216529846
                    'num_samples': int, e.g., 128

                runner.data_loader.dataset
                    results: original data

                runner.data_batch: pipelined data
                    'img_metas': DataContainer, data_cfg
                    'img': tensor
                    'gt_label': tensor
        '''
        bs = runner.outputs['num_samples']
        ans = []
        for i in range(self.repeat): #同一批数据做多次更新
            en_loss, con_loss, cls_loss = 0, 0, 0
            logits_weak = runner.model.module.head.cls_score
            imgs_strong = self.data_aug(runner.data_batch['img_metas'].data[0], self.augment) #搞一张图片出来做增强
            logits_strong = runner.model(img=imgs_strong, return_loss=False, without_softmax=True,post_process=False) #mmcls.models.classifiers.base forward
            if 'entropy' in self.mode:
                en_loss = self.entropy(runner, logits_weak, logits_strong) * self.entropy_weight
                runner.log_buffer.update({'entropy_loss': en_loss.item()})
            if 'cls' in self.mode:
                cls_loss = self.cls(runner, logits_weak, logits_strong) * self.cls_weight
                runner.log_buffer.update({'cls_loss': cls_loss.item()})
            if 'contrast' in self.mode:
                con_loss = self.contrast(runner, logits_strong) * self.contrast_weight
                runner.log_buffer.update({'contrast_loss': con_loss.item()})
            # print(en_loss.item(), cls_loss.item())
            total_loss = en_loss + con_loss + cls_loss
            runner.log_buffer.update({'Total_loss': total_loss.item()})
            self.after_train_iter_optim(runner, total_loss)
 
            # test_accuracy for optimized model
            with torch.no_grad():
                runner.run_iter(runner.data_batch, train_mode=True, **runner.runner_kwargs)
            top1 = runner.outputs['log_vars']['top-1']
            ans.append(top1)

        # print('Iter{}: {}'.format(runner._inner_iter, ans))
        self.num_pos += top1 * bs
        self.num_tot += bs

        self.acc_val = self.num_pos / self.num_tot
        runner.log_buffer.output[self.acc_var] = self.acc_val

    def after_train_epoch(self, runner):
        runner.log_buffer.ready = True
        runner.log_buffer.output[self.total_var] = self.num_tot
        runner.log_buffer.output[self.acc_var] = self.acc_val
        if self.reset:
            runner.model.load_state_dict(self.model_state, strict=True)
            runner.optimizer.load_state_dict(self.optimizer_state)

    @torch.enable_grad()
    def after_train_iter_optim(self, runner, loss):
        runner.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)}, runner.outputs['num_samples'])
        runner.optimizer.step()

    def entropy(self, runner, logits_weak, logits_strong):
        if not hasattr(self, 'augment') or self.img_aug=='weak':
            loss = softmax_entropy(logits_weak).mean(0)
        else:
            if self.entropy_type == 'entropy':
                loss = softmax_entropy(logits_strong).mean(0)
            elif self.entropy_type == 'infomax':
                loss = info_max(logits_strong).mean(0)
            elif self.mode == 'memo':
                imgs = [runner.data_batch['img']]
                for _ in range(1, self.B):
                    img_strong = self.data_aug(runner.data_batch['img_metas'].data[0], self.augment)
                    imgs.append(img_strong)
                imgs = torch.cat(imgs, dim=0)

                outputs = runner.model(img=imgs, return_loss=False, without_softmax=True)
                logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
                avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
                min_real = torch.finfo(avg_logits.dtype).min
                avg_logits = torch.clamp(avg_logits, min=min_real)
                loss = -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)
                
        return loss
    
    def contrast(self, runner, logits_strong):
        loss = getattr(self, self.func)(runner, logits_strong)
        return loss

    def cls(self, runner, logits_weak, logits_strong):
        _, labels_weak = torch.max(logits_weak, dim=1)
        loss = self.cross_entropy_loss(logits_weak, labels_weak, self.ce_type)

        if hasattr(self, 'augment'):
            _, labels_weak = torch.max(logits_strong, dim=1)
            loss_strong = self.cross_entropy_loss(logits_strong, labels_weak, self.ce_type)

            if self.cls_type == 'strong':
                loss = loss_strong
            if self.cls_type == 'both':
                prob_weak = F.softmax(logits_weak, dim=1)
                loss_weak_strong = F.kl_div(F.log_softmax(logits_strong, dim=1), prob_weak, reduction="batchmean")
                loss += loss_weak_strong

        return loss

    def cross_entropy_loss(self, logits, labels, ce_type='standard'):
        target_probs = F.softmax(logits, dim=1)
        if ce_type == "standard":
            return F.cross_entropy(logits, labels)
        elif ce_type == "smoothed":
            epsilon = 0.1
            log_probs = F.log_softmax(logits, dim=1)
            with torch.no_grad():
                targets = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1)
                targets = (1 - epsilon) * targets + epsilon / self.class_num
            loss = (-targets * log_probs).sum(dim=1).mean()
            return loss
        elif ce_type == "soft":
            log_probs = F.log_softmax(logits, dim=1)
            return F.kl_div(log_probs, target_probs, reduction="batchmean")
        
    def configure_samplenorm(self, bs, model):
        """Configure model for use with dent."""
        # disable grad, to (re-)enable only what dent updates
        for m in model.modules():
            if hasattr(m, 'ckpt_weight'):
                if m.weight.requires_grad:
                    m.weight = nn.Parameter(m.ckpt_weight.unsqueeze(0).repeat(bs, 1))
                if m.bias.requires_grad:
                    m.bias = nn.Parameter(m.ckpt_bias.unsqueeze(0).repeat(bs, 1))
        return model

    def init_bank(self, runner):
        if self.queue_size <= 0:
            self.base_sums = torch.zeros(self.projector_dim, self.class_num).to(self.device)
            self.cnt = torch.zeros(self.class_num).to(self.device) + 0.00001
        else:
            # queue_ori
            if self.origin:
                self.queue_ori = nn.Module().to(self.device)
                self.queue_ori.register_buffer("queue_list", torch.randn(self.projector_dim, self.queue_size * self.class_num))
                self.queue_ori.queue_list = F.normalize(self.queue_ori.queue_list, dim=0)
                self.queue_ori.register_buffer("queue_ptr", torch.zeros(self.class_num, dtype=torch.long))
            # queue_all
            if 'all' in self.func:
                self.queue_all = nn.Module().to(self.device)
                self.queue_all.register_buffer("queue_list", torch.randn(self.projector_dim, self.queue_size))
                self.queue_all.queue_list = F.normalize(self.queue_all.queue_list, dim=0)
                self.queue_all.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            else:
                # queue
                self.queue_aug = nn.Module().to(self.device)
                self.queue_aug.register_buffer("queue_list", torch.randn(self.projector_dim, self.queue_size * self.class_num))
                self.queue_aug.queue_list = F.normalize(self.queue_aug.queue_list, dim=0)
                self.queue_aug.register_buffer("queue_ptr", torch.zeros(self.class_num, dtype=torch.long))

    def init_encoder(self, runner):
        # query encoder and key encoder
        self.encoder = build_classifier(self.model_cfg).to(self.device)
        self.encoder = MMDataParallel(self.encoder.to(self.device), device_ids=[self.device])
        for param_q, param_k in zip(runner.model.parameters(), self.encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # don't be updated by gradient

    def data_aug(self, imgs_meta, augment):
        data = []
        for i in range(len(imgs_meta)):
            data.append(augment({'img': imgs_meta[i]['ori_img']})['img'])
        data = torch.stack(data, dim=0)
        return data

    @torch.no_grad()
    def _dequeue_and_enqueue(self, queue, key, labels):
        assert key.size()[0] == len(labels)

        for i in range(len(labels)):
            c = labels[i]
            ptr = int(queue.queue_ptr[c])
            real_ptr = ptr + c * self.queue_size
            queue.queue_list[:, real_ptr:real_ptr + 1] = key[i: i + 1].T
            ptr = (ptr + 1) % self.queue_size  # move pointer
            queue.queue_ptr[c] = ptr
    
    @torch.no_grad()
    def _dequeue_and_enqueue_all(self, queue, key):
        for i in range(key.size()[0]):
            ptr = int(queue.queue_ptr[0])
            queue.queue_list[:, ptr:ptr + 1] = key[i: i + 1].T
            ptr = (ptr + 1) % self.queue_size  # move pointer
            queue.queue_ptr[0] = ptr

    @torch.no_grad()
    def _momentum_update_encoder(self, modelq, modelk):
        # Momentum update of the key encoder
        for param_q, param_k in zip(modelq.parameters(), modelk.parameters()):
            if param_k.data.size() != param_q.data.size():
                continue
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    def pgc(self, runner, logits_strong): 
        """
            Input:
                im_q: a batch of query images
                im_k: a batch of key images
            Output: loss
        """
        imgs_meta = runner.data_batch['img_metas'].data[0]
        img_k = self.data_aug(imgs_meta, self.augment)
        
        q_c = logits_strong
        max_prob, pred_labels = torch.max(q_c, dim=1)

        if self.norm=='L1Norm':
            q_c = F.normalize(q_c, p=1, dim=1)
        elif self.norm=='L2Norm':
            q_c = F.normalize(q_c, p=2, dim=1)
        elif self.norm=='softmax':
            q_c = q_c.softmax(dim=1)
        
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_encoder(runner.model, self.encoder)  # update the key encoder

            k_c = self.encoder(img=img_k, return_loss=False, without_softmax=True)

            if self.norm=='L1Norm':
                k_c = F.normalize(k_c, p=1, dim=1)
            elif self.norm=='L2Norm':
                k_c = F.normalize(k_c, p=2, dim=1)
            elif self.norm=='softmax':
                k_c = k_c.softmax(dim=1)

        # compute logits
        # positive logits: Nx1
        l_pos = torch.einsum('nl,nl->n', [q_c, k_c]).unsqueeze(-1)  # Einstein sum is more intuitive

        # cur_queue_list: queue_size * class_num
        cur_queue_list = self.queue_aug.queue_list.clone().detach()

        # calibration
        if self.projection_expand:
            sampled_data = []
            sampled_label = []
            if not hasattr(self, 'base_means'):
                self.base_sums = [np.zeros(self.projector_dim)] * self.class_num
                self.base_cov = [np.zeros((self.projector_dim, self.projector_dim))] * self.class_num
                self.cnt = [0] * self.class_num

        l_neg_list = torch.Tensor([]).to(self.device)
        l_pos_list = torch.Tensor([]).to(self.device)

        for i in range(q_c.size()[0]):
            neg_sample = torch.cat([cur_queue_list[:, 0:pred_labels[i] * self.queue_size],
                                    cur_queue_list[:, (pred_labels[i] + 1) * self.queue_size:]],
                                dim=1).to(self.device)
            pos_sample = cur_queue_list[:, pred_labels[i] * self.queue_size: (pred_labels[i] + 1) * self.queue_size].to(self.device)
            ith_neg = torch.einsum('nl,lk->nk', [q_c[i: i + 1], neg_sample])
            ith_pos = torch.einsum('nl,lk->nk', [q_c[i: i + 1], pos_sample])
            l_neg_list = torch.cat((l_neg_list, ith_neg), dim=0)
            l_pos_list = torch.cat((l_pos_list, ith_pos), dim=0)

        self._dequeue_and_enqueue(self.queue_aug, k_c, pred_labels)
        # logits: 1 + queue_size + queue_size * (class_num - 1)
        PGC_logits = torch.cat([l_pos, l_pos_list, l_neg_list], dim=1)
        # apply temperature
        PGC_logits = nn.LogSoftmax(dim=1)(PGC_logits / self.temp)

        PGC_labels = torch.zeros([PGC_logits.size()[0], 1 + self.queue_size * self.class_num]).cuda(self.device)
        PGC_labels[:, 0: self.queue_size + 1].fill_(1.0 / (self.queue_size + 1))

        loss = F.kl_div(PGC_logits, PGC_labels, reduction='batchmean')
        return loss

    def bank2PGC(self, runner, logits_strong):
        imgs_meta = runner.data_batch['img_metas'].data[0]
        img_o = self.data_aug(imgs_meta, self.origin)
        img_k = self.data_aug(imgs_meta, self.augment)

        # query: (bs x projector_dim)
        q_c = logits_strong
        q_f = runner.model.module.feat
        # q_f = q_c

        with torch.no_grad():  # no gradient to keys
            # update the key encoder
            self._momentum_update_encoder(runner.model, self.encoder)

            # original img: (bs x projector_dim)
            o_c = self.encoder(img=img_o, return_loss=False, without_softmax=True)
            # o_f = o_c
            o_f = self.encoder.module.feat
            # o_c = runner.model(img=img_o, return_loss=False, without_softmax=True)  
            # o_f = runner.model.module.feat
            # o_f = q_f

            # key: (bs x projector_dim)
            k_c = self.encoder(img=img_k, return_loss=False, without_softmax=True)
            # k_f = k_c
            k_f = self.encoder.module.feat
            # k_c = runner.model(img=img_k, return_loss=False, without_softmax=True)
            # k_f = runner.model.module.feat
            # k_f = q_f
        
        # normalize
        if self.norm=='L1Norm':
            o_f = F.normalize(o_f, p=1, dim=1)
            q_f = F.normalize(q_f, p=1, dim=1)
            k_f = F.normalize(k_f, p=1, dim=1)
        elif self.norm=='L2Norm':
            o_f = F.normalize(o_f, p=2, dim=1)
            q_f = F.normalize(q_f, p=2, dim=1)
            k_f = F.normalize(k_f, p=2, dim=1)
        elif self.norm=='softmax':
            o_f = o_f.softmax(dim=1)
            q_f = q_f.softmax(dim=1)
            k_f = k_f.softmax(dim=1)
        
        # compute logits
        pos_k = torch.einsum('nl,nl->n', [q_f, k_f]).unsqueeze(-1)
        pos_o = torch.einsum('nl,nl->n', [q_f, o_f]).unsqueeze(-1)
        neg_f_ori = self.queue_ori.queue_list.clone().detach().to(self.device)
        neg_f_aug = self.queue_aug.queue_list.clone().detach().to(self.device)
        
        max_pro, pred_labels = torch.max(q_c, dim=1)
        pos_ori, neg_ori = self.get_pgc(o_f, neg_f_ori, pred_labels)
        pos_aug, neg_aug = self.get_pgc(q_f, neg_f_aug, pred_labels)
            
        # logits: 1 + 1 + queue_size + queue_size + 
        #         queue_size * (class_num -1) + queue_size * (class_num -1)
        logits = torch.cat([pos_o, pos_k, pos_ori, pos_aug, neg_ori, neg_aug], dim=1)
        logits = nn.LogSoftmax(dim=1)(logits / self.temp)  # apply temperature

        marks = torch.zeros([logits.size()[0], 2 + 2 * self.queue_size * self.class_num]).cuda(self.device)
        marks[:, : 2 + 2 * self.queue_size].fill_(1.0 / (2 + 2 * self.queue_size))

        loss = F.kl_div(logits, marks, reduction='batchmean')

        # update the queue
        self._dequeue_and_enqueue(self.queue_ori, o_f, pred_labels)
        self._dequeue_and_enqueue(self.queue_aug, k_f, pred_labels)

        return loss

    def get_pgc(self, query, samples, pred_labels):
        l_pos_list = torch.Tensor([]).to(self.device)
        l_neg_list = torch.Tensor([]).to(self.device)
        for i in range(len(pred_labels)):
            c = pred_labels[i]
            pos_sample = samples[:, c * self.queue_size: (c + 1) * self.queue_size].to(self.device)
            neg_sample = torch.cat([samples[:, 0: c * self.queue_size], 
                samples[:, (c + 1) * self.queue_size:]], dim=1).to(self.device)
            ith_pos = torch.einsum('nl,lk->nk', [query[i: i + 1], pos_sample])
            ith_neg = torch.einsum('nl,lk->nk', [query[i: i + 1], neg_sample])
            l_pos_list = torch.cat((l_pos_list, ith_pos), dim=0)
            l_neg_list = torch.cat((l_neg_list, ith_neg), dim=0)
        return l_pos_list, l_neg_list

    def best(self, runner, logits_strong):
        imgs_meta = runner.data_batch['img_metas'].data[0]
        img_k = self.data_aug(imgs_meta, self.augment)

        # query: (bs x projector_dim)
        q_c = logits_strong
        q_f = runner.model.module.feat
        # q_f = q_c

        # with torch.no_grad():  # no gradient to keys
        # key: (bs x projector_dim)
        k_c = runner.model(img=img_k, return_loss=False, without_softmax=True)
        k_f = runner.model.module.feat
        # k_f = k_c
 
        # normalize
        if self.norm=='L1Norm':
            q_f = F.normalize(q_f, p=1, dim=1)
            k_f = F.normalize(k_f, p=1, dim=1)
        elif self.norm=='L2Norm':
            q_f = F.normalize(q_f, p=2, dim=1)
            k_f = F.normalize(k_f, p=2, dim=1)
        elif self.norm=='softmax':
            q_f = q_f.softmax(dim=1)
            k_f = k_f.softmax(dim=1)

        # compute logits
        pos_k = torch.einsum('nl,nl->n', [q_f, k_f]).unsqueeze(-1)
        pos_o = torch.einsum('nl,nl->n', [q_f, q_f]).unsqueeze(-1)
        neg_f_aug = self.queue_aug.queue_list.clone().detach().to(self.device)
        neg_aug = torch.einsum('nl,lk->nk', [q_f, neg_f_aug])
        max_pro, pred_labels = torch.max(q_c, dim=1)
        
        # logits: 1 + 1 + queue_size * class_num
        logits = torch.cat([pos_k, pos_o, neg_aug], dim=1)
        logits = nn.LogSoftmax(dim=1)(logits / self.temp)  # apply temperature

        marks = torch.zeros([logits.size()[0], 2 + self.queue_size * self.class_num]).cuda(self.device)
        marks[:, : 2].fill_(1.0 / 2)

        loss = F.kl_div(logits, marks, reduction='batchmean')

        # update the queue
        self._dequeue_and_enqueue(self.queue_aug, k_f, pred_labels)

        return loss

    def bestAll(self, runner, logits_strong):
        imgs_meta = runner.data_batch['img_metas'].data[0]
        img_k = self.data_aug(imgs_meta, self.augment)

        # query: (bs x projector_dim)
        q_c = logits_strong
        # q_f = q_c
        q_f = runner.model.module.feat

        # with torch.no_grad():  # no gradient to keys
        # key: (bs x projector_dim)
        k_c = runner.model(img=img_k, return_loss=False, without_softmax=True)
        # k_f = k_c
        k_f = runner.model.module.feat
 
        # normalize
        if self.norm=='L1Norm':
            q_f = F.normalize(q_f, p=1, dim=1)
            k_f = F.normalize(k_f, p=1, dim=1)
        elif self.norm=='L2Norm':
            q_f = F.normalize(q_f, p=2, dim=1)
            k_f = F.normalize(k_f, p=2, dim=1)
        elif self.norm=='softmax':
            q_f = q_f.softmax(dim=1)
            k_f = k_f.softmax(dim=1)

        # compute logits
        pos_k = torch.einsum('nl,nl->n', [q_f, k_f]).unsqueeze(-1)
        pos_o = torch.einsum('nl,nl->n', [q_f, q_f]).unsqueeze(-1)
        neg_f_aug = self.queue_all.queue_list.clone().detach().to(self.device)
        neg_aug = torch.einsum('nl,lk->nk', [q_f, neg_f_aug])
        
        # logits: 1 + 1 + queue_size * class_num
        logits = torch.cat([pos_k, pos_o, neg_aug], dim=1)
        logits = nn.LogSoftmax(dim=1)(logits / self.temp)  # apply temperature

        marks = torch.zeros([logits.size()[0], 2 + self.queue_size]).cuda(self.device)
        marks[:, : 2].fill_(1.0 / 2)

        loss = F.kl_div(logits, marks, reduction='batchmean')

        # update the queue
        self._dequeue_and_enqueue_all(self.queue_all, k_f)

        return loss

    def mocoCalib(self, runner, logits_strong):
        imgs_meta = runner.data_batch['img_metas'].data[0]
        img_k = self.data_aug(imgs_meta, self.augment)

        # query: (bs x projector_dim)
        q_c = logits_strong
        # q_f = q_c
        q_f = runner.model.module.feat

        with torch.no_grad():  # no gradient to keys
        #     self._momentum_update_encoder(runner.model, self.encoder)

        #     # key: (bs x projector_dim)
        #     k_c = self.encoder(img=img_k, return_loss=False, without_softmax=True)
        #     # k_f = k_c
        #     k_f = self.encoder.module.feat
            k_c = runner.model(img=img_k, return_loss=False, without_softmax=True)
            # k_f = k_c
            k_f = runner.model.module.feat
 
        # normalize
        if self.norm=='L1Norm':
            q_f = F.normalize(q_f, p=1, dim=1)
            k_f = F.normalize(k_f, p=1, dim=1)
        elif self.norm=='L2Norm':
            q_f = F.normalize(q_f, p=2, dim=1)
            k_f = F.normalize(k_f, p=2, dim=1)
        elif self.norm=='softmax':
            q_f = q_f.softmax(dim=1)
            k_f = k_f.softmax(dim=1)

        # update the calib
        self.calib(k_f, q_c.max(dim=1)[1])

        # compute logits
        pos_k = torch.einsum('nl,nl->n', [q_f, k_f]).unsqueeze(-1)
        neg_f_aug = (self.base_sums / self.cnt).detach().to(self.device)
        neg_aug = torch.einsum('nl,lk->nk', [q_f, neg_f_aug])
        
        # logits: 1 + 1 + queue_size * class_num
        logits = torch.cat([pos_k, neg_aug], dim=1)
        logits = nn.LogSoftmax(dim=1)(logits / self.temp)  # apply temperature

        marks = torch.zeros([logits.size()[0], 1 + self.class_num]).cuda(self.device)
        marks[:, : 1].fill_(1.0 / 1)

        loss = F.kl_div(logits, marks, reduction='batchmean')

        return loss
 
    def calib(self, logits, labels):
        if self.CLUE:
            prob = F.softmax(logits, dim=1)
            log_prob = F.log_softmax(logits, dim=1)
            entropy = -torch.einsum('nl,nl->n', [prob, log_prob])
        for i in range(len(labels)):
            c = labels[i]
            if self.CLUE:
                self.base_sums[:, c] += entropy[i] * logits[i, :].T
                self.cnt[c] += entropy[i]
            else:
                self.base_sums[:, c] += logits[i, :].T
                self.cnt[c] += 1