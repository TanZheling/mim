import torch
from torch import nn
import torch.nn.functional as F
import copy
import numpy as np
from mmcv.runner import HOOKS, Fp16OptimizerHook, build_optimizer, LossScaler, wrap_fp16_model
#from mmcv.ops import info_max
from mmcls.datasets.pipelines import Compose
from mmcls.models import build_classifier
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
import torch.jit

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

# used as an optimizer hook
@HOOKS.register_module()
class TentOptimizerHook(Fp16OptimizerHook):
    def __init__(self,
                 optimizer_cfg=None,
                 loss_cfg=None,  # type in ['kl_div', 'infomax', 'entropy', 'PGC']
                 augment_cfg=None,
                 reset=None,  # [None, 'batch', 'sample']
                 repeat=1,
                 sampleAware=False,
                 fp16=None,
                 grad_clip=None):
        self.optimizer_cfg = optimizer_cfg
        self.loss_cfg = loss_cfg if loss_cfg!=None else dict(type='entropy')
        self.reset = reset
        self.repeat = repeat
        self.sampleAware = sampleAware
        self.mode = self.loss_cfg['mode']
        self.fp16 = fp16
        self.grad_clip = grad_clip
        if self.fp16:
            super().__init__(**fp16)
        
        assert repeat > 0

        self.origin = Compose(self.loss_cfg.origin_pipeline) if 'origin_pipeline' in self.loss_cfg else None
        self.bottomup = self.loss_cfg.get('bottomup', [-1])
        assert isinstance(self.bottomup, list)

        if 'entropy' in self.mode:
            self.entropy_weight = self.loss_cfg.get('entropy_weight', 1)
            self.entropy_type = self.loss_cfg.get('entropy_type', 'entropy')
            self.img_aug = self.loss_cfg.get('img_aug', 'weak')
            self.B = self.loss_cfg.get('B', 1)

        if 'contrast' in self.mode:
            self.model_cfg = self.loss_cfg.model_cfg
            self.contrast_weight = self.loss_cfg.get('contrast_weight', 1)
            self.projector_dim = self.loss_cfg.get('projector_dim', [10])
            self.class_num = self.loss_cfg.get('class_num', 100)
            self.queue_size = self.loss_cfg.get('queue_size', 1)
            self.norm = self.loss_cfg.get('norm', 'L1Norm')
            self.temp = self.loss_cfg.get('temp', 0.07)
            self.momentum = 0.999
            self.pool = self.loss_cfg.get('pool', 'avg')

            self.func = self.loss_cfg.get('func', 'best')
            self.CLUE = self.loss_cfg.get('CLUE', False)
            self.alpha = self.loss_cfg.get('alpha', 0.21)
            self.projection_expand = self.loss_cfg.get('projection_expand_num', 0)
            self.feature_expand = self.loss_cfg.get('feature_expand_num', 0)

        if 'cls' in self.mode:
            self.cls_weight = self.loss_cfg.get('cls_weight', 1)
            self.cls_type = self.loss_cfg.get('cls_type', 'weak')
            self.ce_type = self.loss_cfg.get('ce_type', 'smoothed')
            self.class_num = self.loss_cfg.get('class_num', 100)

        if augment_cfg is not None:
            print("Generating the augmentation.")
            self.augment = Compose(augment_cfg)

        if self.fp16:
            torch.set_default_tensor_type(torch.HalfTensor)

    def before_run(self, runner):
        self.device = runner.model.device_ids[0]
        self.device_list = runner.model.device_ids
        if 'contrast' in self.mode:
            self.init_encoder(runner)
            self.init_bank(runner)

        if self.fp16:
            """Preparing steps before Mixed Precision Training."""
            # wrap model mode to fp16
            wrap_fp16_model(runner.model)
            if 'contrast' in self.mode:
                wrap_fp16_model(self.encoder)
            # resume from state dict
            if 'fp16' in runner.meta and 'loss_scaler' in runner.meta['fp16']:
                scaler_state_dict = runner.meta['fp16']['loss_scaler']
                self.loss_scaler.load_state_dict(scaler_state_dict)

        if self.reset:
            runner.logger.info("Storing the related states.")
            self.model_state = copy.deepcopy(runner.model.state_dict())
            self.optimizer_state = copy.deepcopy(runner.optimizer.state_dict())

    def before_train_epoch(self, runner):
        corr = runner.data_loader.dataset.corruption
        sev = runner.data_loader.dataset.severity
        if len(corr) > 1:
            self.acc_var = 'multi/online_accuracy_top-1'
            self.acc_var_aug = 'multi/online_accuracy_top-1_aug'
            self.total_var = 'multi/online_total_num'
        else:
            self.acc_var = str(sev[0]) + '/online_accuracy_top-1'
            self.acc_var_aug = str(sev[0]) + '/online_accuracy_top-1_aug'
            self.total_var = str(sev[0]) + '/online_total_num'
        self.num_pos, self.num_tot = 0, 0
        self.num_pos_aug = 0

    def before_train_iter(self, runner):
        if self.reset:
            runner.model.load_state_dict(self.model_state, strict=True)
            if 'contrast' in self.mode:
                self.encoder.load_state_dict(self.model_state, strict=True)
            runner.optimizer.load_state_dict(self.optimizer_state)

        if self.reset == 'sample' or self.sampleAware:
            runner.model = self.configure_norm(runner.data_batch['img'].size(0), runner.model)
            if 'contrast' in self.mode:
                self.encoder = self.configure_norm(runner.data_batch['img'].size(0), self.encoder)
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
        top1_ori = runner.outputs['log_vars']['top-1']
        if torch.distributed.is_initialized():
            bs *= torch.distributed.get_world_size()
        for i in range(self.repeat):
            for stage in self.bottomup:
                if True:
                    imgs_strong = self.data_aug(runner.data_batch['img_metas'].data[0], self.augment)
                    break
                self.stage = stage
                en_loss, con_loss, cls_loss = 0, 0, 0
                self.logits_weak = runner.model.module.head.cls_score
                self.feats_weak = runner.model.module.feat
                imgs_strong = self.data_aug(runner.data_batch['img_metas'].data[0], self.augment)
                self.logits_strong = runner.model(img=imgs_strong, return_loss=False, without_softmax=True)
                self.feats_strong = runner.model.module.feat
                self.medium_level = runner.model.module.medium_level
                if 'entropy' in self.mode:
                    en_loss = self.entropy(runner) * self.entropy_weight
                    runner.log_buffer.update({'entropy_loss': en_loss.item()})
                if 'cls' in self.mode:
                    cls_loss = self.cls(runner) * self.cls_weight
                    runner.log_buffer.update({'cls_loss': cls_loss.item()})
                if 'contrast' in self.mode:
                    con_loss = self.contrast(runner) * self.contrast_weight
                    runner.log_buffer.update({'contrast_loss': con_loss.item()})
                # print(en_loss.item(), cls_loss.item())
                total_loss = en_loss + con_loss + cls_loss
                runner.log_buffer.update({'Total_loss': total_loss.item()})
                if self.fp16:
                    self.after_train_iter_optim_fp16(runner, total_loss)
                else:
                    self.after_train_iter_optim(runner, total_loss)

            self.after_train_iter_optim(runner, runner.outputs['loss'])
            # test_accuracy for optimized model
            with torch.no_grad():
                # runner.data_batch['img'] = self.data_aug(runner.data_batch['img_metas'].data[0], self.origin)
                # runner.run_iter(runner.data_batch, train_mode=True, **runner.runner_kwargs)
                # top1_ori = runner.outputs['log_vars']['top-1']
                runner.data_batch['img'] = imgs_strong
                print((runner))
                runner.run_iter(runner.data_batch, train_mode=True, **runner.runner_kwargs)
                top1_aug = runner.outputs['log_vars']['top-1']

        # print('Iter{}: {}'.format(runner._inner_iter, ans))
        self.num_pos += top1_ori * bs
        self.num_tot += bs
        self.num_pos_aug += top1_aug * bs

        self.acc_val = self.num_pos / self.num_tot
        runner.log_buffer.output[self.acc_var] = self.acc_val
        self.acc_val_aug = self.num_pos_aug / self.num_tot
        runner.log_buffer.output[self.acc_var_aug] = self.acc_val_aug

        if self.sampleAware and not self.reset:
            runner.model = self.configure_norm(runner.data_batch['img'].size(0), runner.model, direction='SampleToBatch')
            if 'contrast' in self.mode:
                self.encoder = self.configure_norm(runner.data_batch['img'].size(0), self.encoder, direction='SampleToBatch')

    def after_train_epoch(self, runner):
        runner.log_buffer.ready = True
        runner.log_buffer.output[self.total_var] = self.num_tot
        runner.log_buffer.output[self.acc_var] = self.acc_val
        runner.log_buffer.output[self.acc_var_aug] = self.acc_val_aug
        if self.reset:
            runner.model.load_state_dict(self.model_state, strict=True)
            runner.optimizer.load_state_dict(self.optimizer_state)

    def after_train_iter_optim(self, runner, loss):
        runner.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)}, runner.outputs['num_samples'])
        runner.optimizer.step()

    def after_train_iter_optim_fp16(self, runner, loss):
        """Backward optimization steps for Mixed Precision Training. For
            dynamic loss scaling, please refer to
            https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.
            1. Scale the loss by a scale factor.
            2. Backward the loss to obtain the gradients.
            3. Unscale the optimizer’s gradient tensors.
            4. Call optimizer.step() and update scale factor.
            5. Save loss_scaler state_dict for resume purpose.
        """
        # clear grads of last iteration
        runner.model.zero_grad()
        runner.optimizer.zero_grad()

        self.loss_scaler.scale(loss).backward()
        self.loss_scaler.unscale_(runner.optimizer)
        # grad clip
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                            runner.outputs['num_samples'])
        # backward and update scaler
        self.loss_scaler.step(runner.optimizer)
        self.loss_scaler.update(self._scale_update_param)

        # save state_dict of loss_scaler
        runner.meta.setdefault(
            'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()

    def entropy(self, runner):
        if not hasattr(self, 'augment') or self.img_aug=='weak':
            loss = softmax_entropy(self.logits_weak).mean(0)
        else:
            if self.entropy_type == 'entropy':
                loss = softmax_entropy(self.logits_strong).mean(0)
            #elif self.entropy_type == 'infomax':
            #    loss = info_max(self.logits_strong).mean(0)
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
    
    def contrast(self, runner):
        loss = getattr(self, self.func)(runner)
        return loss

    def cls(self, runner):
        _, labels_weak = torch.max(self.logits_weak, dim=1)
        loss = self.cross_entropy_loss(self.logits_weak, labels_weak, self.ce_type)

        if hasattr(self, 'augment'):
            _, labels_weak = torch.max(self.logits_strong, dim=1)
            loss_strong = self.cross_entropy_loss(self.logits_strong, labels_weak, self.ce_type)

            if self.cls_type == 'strong':
                loss = loss_strong
            if self.cls_type == 'both':
                prob_weak = F.softmax(self.logits_weak, dim=1)
                loss_weak_strong = F.kl_div(F.log_softmax(self.logits_strong, dim=1), prob_weak, reduction="batchmean")
                loss += loss_weak_strong

        return loss

    @torch.no_grad()
    def all_gather(self, x, mode='cat', hold_type=True):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        args:
            mode: default='cat', include ['cat', 'sum', 'mean', 'default']
        """
        x_type = type(x)
        tensor = torch.tensor(x, dtype=torch.float64)
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        if mode == 'cat':
            output = torch.cat(tensors_gather, dim=0)
        elif mode == 'sum':
            output = sum(tensors_gather)
        elif mode == 'mean':
            output = sum(tensors_gather) / torch.distributed.get_world_size()
        return x_type(output) if hold_type else output

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
        
    def configure_norm(self, bs, model, direction='BatchToSample'):
        """Configure model for use with dent."""
        # disable grad, to (re-)enable only what dent updates
        if direction == 'BatchToSample':
            for m in model.modules():
                if hasattr(m, 'ckpt_weight'):
                    if m.weight.requires_grad:
                        m.weight = nn.Parameter(m.ckpt_weight.unsqueeze(0).repeat(bs, 1))
                    if m.bias.requires_grad:
                        m.bias = nn.Parameter(m.ckpt_bias.unsqueeze(0).repeat(bs, 1))
        elif direction == 'SampleToBatch':
            for m in model.modules():
                if hasattr(m, 'ckpt_weight'):
                    if m.weight.requires_grad:
                        m.ckpt_weight = m.weight.mean(dim=0)
                        m.weight = nn.Parameter(m.ckpt_weight)  # for offline testing
                    if m.bias.requires_grad:
                        m.ckpt_bias = m.bias.mean(dim=0)
                        m.bias = nn.Parameter(m.ckpt_bias)  # for offline testing
        return model

    def init_bank(self, runner):
        if self.queue_size <= 0:
            self.base_sums, self.cnt = [], []
            for stage in self.bottomup:
                self.base_sums.append(torch.zeros(self.projector_dim[stage], self.class_num).to(self.device))
                self.cnt.append(torch.zeros(self.class_num).to(self.device) + 0.00001)
        else:
            # queue_ori
            if 'banko' in self.func or 'bank2' in self.func:
                self.queue_ori = nn.Module().to(self.device)
                self.queue_ori.register_buffer("queue_list", torch.randn(self.projector_dim, self.queue_size * self.class_num, dtype=torch.float))
                self.queue_ori.queue_list = F.normalize(self.queue_ori.queue_list, dim=0)
                self.queue_ori.register_buffer("queue_ptr", torch.zeros(self.class_num, dtype=torch.long))
            # queue_all
            if 'All' in self.func:
                self.queue_all = []
                for stage in self.bottomup:
                    tmp = nn.Module().to(self.device)
                    tmp.register_buffer("queue_list", torch.randn(self.projector_dim[stage], self.queue_size, dtype=torch.float))
                    tmp.queue_list = F.normalize(tmp.queue_list, dim=0)
                    tmp.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
                    self.queue_all.append(tmp)
            else:
                # queue
                self.queue_aug = nn.Module().to(self.device)
                self.queue_aug.register_buffer("queue_list", torch.randn(self.projector_dim, self.queue_size * self.class_num, dtype=torch.float))
                self.queue_aug.queue_list = F.normalize(self.queue_aug.queue_list, dim=0)
                self.queue_aug.register_buffer("queue_ptr", torch.zeros(self.class_num, dtype=torch.long))

    def init_encoder(self, runner):
        if self.fp16:
            torch.set_default_tensor_type(torch.FloatTensor)
        # query encoder and key encoder
        if torch.distributed.is_initialized():
            self.encoder = MMDistributedDataParallel(
                build_classifier(self.model_cfg).cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=False)
        else:
            self.encoder = MMDataParallel(build_classifier(self.model_cfg).cuda(self.device), device_ids=self.device_list)

        for param_q, param_k in zip(runner.model.parameters(), self.encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # don't be updated by gradient
        if self.fp16:
            torch.set_default_tensor_type(torch.HalfTensor)

    def data_aug(self, imgs_meta, augment):
        data = []
        for i in range(len(imgs_meta)):
            data.append(augment({'img': imgs_meta[i]['ori_img']})['img'])
        data = torch.stack(data, dim=0)
        # if self.fp16:
        #     data = data.to(torch.half)
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

    def distribution_calibration(self, query, base_means, base_cov, k):
        dist = []
        for i in range(len(base_means)):
            dist.append(np.linalg.norm(query-base_means[i]))
        index = np.argpartition(dist, k)[:k]
        mean = np.concatenate([np.array(base_means)[index], query[np.newaxis, :]])
        calibrated_mean = np.mean(mean, axis=0)
        calibrated_cov = np.mean(np.array(base_cov)[index], axis=0) + self.alpha
        for i in range(k):
            c = index[i]
            if self.CLUE:
                softmax = np.exp(query)/np.sum(np.exp(query))
                entropy = (-softmax * np.log(softmax)).sum()
                self.base_sums[c] = self.base_sums[c] + entropy
                self.cnt[c] += entropy
            else:
                self.base_sums[c] = self.base_sums[c] + query
                self.cnt[c] += 1
        return calibrated_mean, calibrated_cov

    def feature_calibration(self, im_q, im_k, labels, k=2):
        bs = im_q.size()[0]
        for i in range(bs):
            self.feature_base_means[labels[i]] *= self.feature_base_cnt
            self.feature_base_cnt += 1
            self.feature_base_means[labels[i]] /= self.feature_base_cnt
        for i in range(bs):
            mean, cov = self.distribution_calibration(
                img_q[i].cpu().detach().numpy(), 
                self.feature_base_means, 
                self.feature_base_cov,
                k=k)
            new_features = np.random.uniform(mean-0.01, mean+0.01, size=self.feature_expand)
            img_q.concatenate([img_q, new_features], dim=0)
            img_k.concatenate([img_k] + [img_k[i]]*self.feature_expand, dim=0)
        return img_q.reshape(im_q.size()), img_k

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
            if self.fp16:
                neg_sample = neg_sample.to(torch.half)
                pos_sample = pos_sample.to(torch.half)
            ith_neg = torch.einsum('nl,lk->nk', [q_c[i: i + 1], neg_sample])
            ith_pos = torch.einsum('nl,lk->nk', [q_c[i: i + 1], pos_sample])
            l_neg_list = torch.cat((l_neg_list, ith_neg), dim=0)
            l_pos_list = torch.cat((l_pos_list, ith_pos), dim=0)
            # self._dequeue_and_enqueue(self.queue_aug, k_c[i: i + 1], pred_labels[i])
            if self.projection_expand:
                sample = q_c[i].cpu().detach().numpy()
                base_mean = [self.base_sums[i]/(self.cnt[i] if self.cnt[i] else 1) for i in range(self.class_num)]
                mean, cov = self.distribution_calibration(sample, base_mean, self.base_cov, k=1)
                sampled_data.append(np.random.multivariate_normal(mean=mean, cov=cov, size=self.projection_expand))
                # sampled_data.append(np.random.uniform(mean-0.001, mean+0.001, size=(self.projection_expand, self.projector_dim)))
                sampled_label.extend([pred_labels[i]] * self.projection_expand)   

        if self.projection_expand:
            sampled_data = torch.tensor(np.concatenate(sampled_data)).to(torch.float32).to(self.device)
            l_pos = torch.cat([l_pos] * (1 + self.projection_expand), dim=0)
            for i in range(batch_size * self.projection_expand):
                neg_sample = torch.cat([cur_queue_list[:, 0:sampled_label[i] * self.queue_size],
                                        cur_queue_list[:, (sampled_label[i] + 1) * self.queue_size:]],
                                    dim=1).to(self.device)
                pos_sample = cur_queue_list[:, sampled_label[i] * self.queue_size: (sampled_label[i] + 1) * self.queue_size].to(self.device)
                ith_neg = torch.einsum('nl,lk->nk', [sampled_data[i: i + 1], neg_sample])
                ith_pos = torch.einsum('nl,lk->nk', [sampled_data[i: i + 1], pos_sample])
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

    def bankoMoco(self, runner):
        imgs_meta = runner.data_batch['img_metas'].data[0]
        img_o = self.data_aug(imgs_meta, self.origin)
        img_k = self.data_aug(imgs_meta, self.augment)

        # query: c: (bs x class_num); f: (bs x projector_dim)
        q_c = runner.model.module.head.cls_score
        q_f = runner.model.module.feat

        with torch.no_grad():  # no gradient to keys
            # update the key encoder
            self._momentum_update_encoder(runner.model, self.encoder)

            # origin: (bs x projector_dim)
            # o_c = runner.model(img=img_o, return_loss=False, without_softmax=True)
            # o_f = runner.model.module.feat
            o_f = q_f

            # key: (bs x projector_dim)
            k_c = self.encoder(img=img_k, return_loss=False, without_softmax=True)
            k_f = self.encoder.module.feat
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
        pos = torch.einsum('nl,nl->n', [q_f, o_f]).unsqueeze(-1)
        pos_k = torch.einsum('nl,nl->n', [q_f, k_f]).unsqueeze(-1)
        neg_f_ori = self.queue_ori.queue_list.clone().detach().to(self.device)
        neg_ori = torch.einsum('nl,lk->nk', [q_f, neg_f_ori])
        max_pro, pred_labels = torch.max(q_c, dim=1)
        
        # logits: 1 + queue_size * class_num
        # logits = torch.cat([pos, neg_ori], dim=1)
        # logits = nn.LogSoftmax(dim=1)(logits / self.temp)  # apply temperature

        # marks = torch.zeros([logits.size()[0], 1 + self.queue_size * self.class_num]).cuda(self.device)
        # marks[:, : 1].fill_(1.0 / (1))

        # logits: 1 + 1 + queue_size * class_num
        logits = torch.cat([pos, pos_k, neg_ori], dim=1)
        logits = nn.LogSoftmax(dim=1)(logits / self.temp)  # apply temperature

        marks = torch.zeros([logits.size()[0], 1 + 1 + self.queue_size * self.class_num]).cuda(self.device)
        marks[:, : 2].fill_(1.0 / (2))

        loss = F.kl_div(logits, marks, reduction='batchmean')

        # update the queue
        self._dequeue_and_enqueue(self.queue_ori, o_f, pred_labels)

        return loss

    def bankoPGC(self, runner):
        imgs_meta = runner.data_batch['img_metas'].data[0]
        img_o = self.data_aug(imgs_meta, self.origin)
        img_k = self.data_aug(imgs_meta, self.augment)

        # query: (bs x projector_dim)
        q_c = runner.model.module.head.cls_score
        q_f = runner.model.module.feat

        with torch.no_grad():  # no gradient to keys
            # update the key encoder
            self._momentum_update_encoder(runner.model, self.encoder)

            # key: (bs x projector_dim)
            # o_c = runner.model(img=img_o, return_loss=False, without_softmax=True)
            # o_f = runner.model.module.feat
            o_f = q_f

            # key: (bs x projector_dim)
            k_c = self.encoder(img=img_k, return_loss=False, without_softmax=True)
            k_f = self.encoder.module.feat
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
        pos = torch.einsum('nl,nl->n', [q_f, o_f]).unsqueeze(-1)
        pos_k = torch.einsum('nl,nl->n', [q_f, k_f]).unsqueeze(-1)
        neg_f_ori = self.queue_ori.queue_list.clone().detach().to(self.device)
        
        max_pro, pred_labels = torch.max(q_c, dim=1)
        pos_ori, neg_ori = self.get_pgc(q_f, neg_f_ori, pred_labels)
            
        # logits: 1 + 1 + queue_size + queue_size * (class_num - 1) 
        logits = torch.cat([pos, pos_k, pos_ori, neg_ori], dim=1)
        logits = nn.LogSoftmax(dim=1)(logits / self.temp)  # apply temperature

        marks = torch.zeros([logits.size()[0], 1 + 1 + self.queue_size * self.class_num]).cuda(self.device)
        marks[:, : 2 + self.queue_size].fill_(1.0 / (2 + self.queue_size))

        loss = F.kl_div(logits, marks, reduction='batchmean')

        # update the queue
        self._dequeue_and_enqueue(self.queue_ori, o_f, pred_labels)

        return loss

    def bankkMoco(self, runner):
        imgs_meta = runner.data_batch['img_metas'].data[0]
        img_o = self.data_aug(imgs_meta, self.origin)
        img_k = self.data_aug(imgs_meta, self.augment)

        # query: (bs x projector_dim)
        q_c = runner.model.module.head.cls_score
        q_f = runner.model.module.feat

        with torch.no_grad():  # no gradient to keys
            # update the key encoder
            self._momentum_update_encoder(runner.model, self.encoder)

            # key: (bs x projector_dim)
            # k_c = runner.model(img=img_k, return_loss=False, without_softmax=True)
            # k_f = runner.model.module.feat
            k_f = q_f

            # origin: (bs x projector_dim)
            o_c = self.encoder(img=img_o, return_loss=False, without_softmax=True)
            o_f = self.encoder.module.feat
            # o_f = q_f

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
        pos = torch.einsum('nl,nl->n', [q_f, k_f]).unsqueeze(-1)
        pos_o = torch.einsum('nl,nl->n', [q_f, o_f]).unsqueeze(-1)
        neg_f_aug = self.queue_aug.queue_list.clone().detach().to(self.device)
        neg_aug = torch.einsum('nl,lk->nk', [q_f, neg_f_aug])
        max_pro, pred_labels = torch.max(q_c, dim=1)
        
        # logits: 1 + 1 + queue_size * class_num
        logits = torch.cat([pos, pos_o, neg_aug], dim=1)
        logits = nn.LogSoftmax(dim=1)(logits / self.temp)  # apply temperature

        marks = torch.zeros([logits.size()[0], 1 + 1 + self.queue_size * self.class_num]).cuda(self.device)
        marks[:, : 2].fill_(1.0 / (2))

        loss = F.kl_div(logits, marks, reduction='batchmean')

        # update the queue
        self._dequeue_and_enqueue(self.queue_aug, k_f, pred_labels)

        return loss

    def bankkPGC(self, runner):
        imgs_meta = runner.data_batch['img_metas'].data[0]
        # img_o = self.data_aug(imgs_meta, self.origin)
        img_k = self.data_aug(imgs_meta, self.augment)

        # query: (bs x projector_dim)
        q_c = runner.model.module.head.cls_score
        # q_f = runner.model.module.feat
        q_f = q_c

        with torch.no_grad():  # no gradient to keys
            # update the key encoder
            self._momentum_update_encoder(runner.model, self.encoder)

            # key: (bs x projector_dim)
            k_c = self.encoder(img=img_k, return_loss=False, without_softmax=True)
            # k_f = self.encoder.module.feat
            k_f = q_c

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
        pos = torch.einsum('nl,nl->n', [q_f, k_f]).unsqueeze(-1)
        neg_f_aug = self.queue_aug.queue_list.clone().detach().to(self.device)
        
        max_pro, pred_labels = torch.max(q_c, dim=1)
        pos_aug, neg_aug = self.get_pgc(q_f, neg_f_aug, pred_labels)
            
        # logits: 1 + 1 + queue_size + queue_size * (class_num -1)
        logits = torch.cat([pos, pos_aug, neg_aug], dim=1)
        logits = nn.LogSoftmax(dim=1)(logits / self.temp)  # apply temperature

        marks = torch.zeros([logits.size()[0], 1 + self.queue_size * self.class_num]).cuda(self.device)
        marks[:, : 1 + self.queue_size].fill_(1.0 / (1 + self.queue_size))

        loss = F.kl_div(logits, marks, reduction='batchmean')

        # update the queue
        self._dequeue_and_enqueue(self.queue_aug, k_f, pred_labels)

        return loss

    def bank2Moco(self, runner): 
        '''
            _o: original
            _q: query
            _k: key
            _c: classification
            _f: feature
        '''
        imgs_meta = runner.data_batch['img_metas'].data[0]
        img_o = self.data_aug(imgs_meta, self.origin)
        img_k = self.data_aug(imgs_meta, self.augment)

        # query: (bs x projector_dim)
        q_c = runner.model.module.head.cls_score
        q_f = runner.model.module.feat

        with torch.no_grad():  # no gradient to keys
            # update the key encoder
            self._momentum_update_encoder(runner.model, self.encoder)

            # original img: (bs x projector_dim)
            o_c = self.encoder(img=img_o, return_loss=False, without_softmax=True)  
            o_f = self.encoder.module.feat
            # o_c = runner.model(img=img_o, return_loss=False, without_softmax=True)  
            # o_f = runner.model.module.feat
            # o_f = q_f

            # key: (bs x projector_dim)
            k_c = self.encoder(img=img_k, return_loss=False, without_softmax=True)
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
        neg_ori = torch.einsum('nl,lk->nk', [q_f, neg_f_ori])
        neg_aug = torch.einsum('nl,lk->nk', [q_f, neg_f_aug])
        
        max_pro, pred_labels = torch.max(q_c, dim=1)
            
        # logits: 1 + 1 + queue_size * class_num + queue_size * class_num
        logits = torch.cat([pos_k, pos_o, neg_ori, neg_aug], dim=1)
        logits = nn.LogSoftmax(dim=1)(logits / self.temp)  # apply temperature

        marks = torch.zeros([logits.size()[0], 2 + 2 * self.queue_size * self.class_num]).cuda(self.device)
        marks[:, : 2].fill_(1.0 / (2))

        loss = F.kl_div(logits, marks, reduction='batchmean')

        # update the queue
        self._dequeue_and_enqueue(self.queue_ori, o_f, pred_labels)
        self._dequeue_and_enqueue(self.queue_aug, k_f, pred_labels)

        return loss

    def bank2PGC(self, runner):
        imgs_meta = runner.data_batch['img_metas'].data[0]
        img_o = self.data_aug(imgs_meta, self.origin)
        img_k = self.data_aug(imgs_meta, self.augment)

        # query: (bs x projector_dim)
        q_c = self.logits_strong
        q_f = self.feats_strong
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

    def best(self, runner):
        imgs_meta = runner.data_batch['img_metas'].data[0]
        img_k = self.data_aug(imgs_meta, self.augment)

        # query: (bs x projector_dim)
        q_c = self.logits_strong
        q_f = self.feats_strong
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
        if self.fp16:
            neg_f_aug = neg_f_aug.to(torch.half)
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

    def bestAll(self, runner):
        imgs_meta = runner.data_batch['img_metas'].data[0]
        img_k = self.data_aug(imgs_meta, self.augment)
        ith_stage = self.bottomup.index(self.stage)

        # query: (bs x projector_dim)
        q_c = self.logits_strong
        # q_f = q_c
        q_f = self.medium_level[self.stage]

        # with torch.no_grad():  # no gradient to keys
        # key: (bs x projector_dim)
        k_c = runner.model(img=img_k, return_loss=False, without_softmax=True)
        # k_f = k_c
        k_f = runner.model.module.medium_level[self.stage]
 
        if len(q_f.size()) > 2:
            if self.pool == 'avg':
                pool = nn.AvgPool2d(kernel_size=q_f.size()[-1])
            elif self.pool == 'max':
                pool = nn.MaxPool2d(kernel_size=q_f.size()[-1])
            q_f = pool(q_f).squeeze().squeeze()
            k_f = pool(k_f).squeeze().squeeze()

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
        elif self.norm=='entropy':
            q_f = -(q_f.softmax(1) * q_f.log_softmax(1))
            k_f = -(k_f.softmax(1) * k_f.log_softmax(1))


        # compute logits
        neg_f_aug = self.queue_all[ith_stage].queue_list.clone().detach().to(self.device)
        if self.fp16:
            neg_f_aug = neg_f_aug.to(torch.half)
            q_f = q_f.to(torch.half)
            k_f = k_f.to(torch.half)
        pos_k = torch.einsum('nl,nl->n', [q_f, k_f]).unsqueeze(-1)
        pos_o = torch.einsum('nl,nl->n', [q_f, q_f]).unsqueeze(-1)
        
        neg_aug = torch.einsum('nl,lk->nk', [q_f, neg_f_aug])
        
        # logits: 1 + 1 + queue_size * class_num
        logits = torch.cat([pos_k, pos_o, neg_aug], dim=1)
        logits = nn.LogSoftmax(dim=1)(logits / self.temp)  # apply temperature

        marks = torch.zeros([logits.size()[0], 2 + self.queue_size]).cuda(self.device)
        marks[:, : 2].fill_(1.0 / 2)

        loss = F.kl_div(logits, marks, reduction='batchmean')

        # update the queue
        self._dequeue_and_enqueue_all(self.queue_all[ith_stage], k_f)

        return loss

    def mocoCalib(self, runner):
        imgs_meta = runner.data_batch['img_metas'].data[0]
        img_k = self.data_aug(imgs_meta, self.augment)

        # query: (bs x projector_dim)
        q_c = self.logits_strong
        q_f = self.medium_level[self.stage]
        
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_encoder(runner.model, self.encoder)

            k_c = self.encoder(img=img_k, return_loss=False, without_softmax=True)
            k_f = self.encoder.module.medium_level[self.stage]

        if len(q_f.size()) > 2:
            if self.pool == 'avg':
                pool = nn.AvgPool2d(kernel_size=q_f.size()[-1])
            elif self.pool == 'max':
                pool = nn.MaxPool2d(kernel_size=q_f.size()[-1])
            q_f = pool(q_f).squeeze()
            k_f = pool(k_f).squeeze()

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
        elif self.norm=='entropy':
            q_f = -(q_f.softmax(1) * q_f.log_softmax(1))
            k_f = -(k_f.softmax(1) * k_f.log_softmax(1))

        # update the calib
        ith_stage = self.bottomup.index(self.stage)
        self.calib(ith_stage, k_f, q_c.max(dim=1)[1])

        # compute logits
        pos_k = torch.einsum('nl,nl->n', [q_f, k_f]).unsqueeze(-1)
        neg_f_aug = (self.base_sums[ith_stage] / self.cnt[ith_stage]).detach().to(self.device)
        neg_aug = torch.einsum('nl,lk->nk', [q_f, neg_f_aug])
        
        # logits: 1 + queue_size * class_num
        logits = torch.cat([pos_k, neg_aug], dim=1)
        logits = nn.LogSoftmax(dim=1)(logits / self.temp)  # apply temperature

        marks = torch.zeros([logits.size()[0], 1 + self.class_num]).cuda(self.device)
        marks[:, : 1].fill_(1.0 / 1)

        loss = F.kl_div(logits, marks, reduction='batchmean')

        return loss

    def calib(self, stage, logits, labels):
        if self.CLUE:
            prob = F.softmax(logits, dim=1)
            log_prob = F.log_softmax(logits, dim=1)
            entropy = -torch.einsum('nl,nl->n', [prob, log_prob])
        for i in range(len(labels)):
            c = labels[i]
            if self.CLUE:
                self.base_sums[stage][:, c] += entropy[i] * logits[i, :].T
                self.cnt[stage][c] += entropy[i]
            else:
                self.base_sums[stage][:, c] += logits[i, :].T
                self.cnt[stage][c] += 1

    def mocoAll(self, runner):
        imgs_meta = runner.data_batch['img_metas'].data[0]
        img_k = self.data_aug(imgs_meta, self.augment)

        # query: (bs x projector_dim)
        q_c = self.logits_strong
        # q_f = q_c
        q_f = self.feats_strong

        with torch.no_grad():  # no gradient to keys
            self._momentum_update_encoder(runner.model, self.encoder)

            # key: (bs x projector_dim)
            k_c = self.encoder(img=img_k, return_loss=False, without_softmax=True)
            # k_f = k_c
            k_f = self.encoder.module.feat
 
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
        neg_f_aug = self.queue_all.queue_list.clone().detach().to(self.device)
        neg_aug = torch.einsum('nl,lk->nk', [q_f, neg_f_aug])
        
        # logits: 1 + 1 + queue_size * class_num
        logits = torch.cat([pos_k, neg_aug], dim=1)
        logits = nn.LogSoftmax(dim=1)(logits / self.temp)  # apply temperature

        marks = torch.zeros([logits.size()[0], 1 + self.queue_size]).cuda(self.device)
        marks[:, : 1].fill_(1.0 / 1)

        loss = F.kl_div(logits, marks, reduction='batchmean')

        # update the queue
        self._dequeue_and_enqueue_all(self.queue_all, k_f)

        return loss

# used as an optimizer hook when dustributed
class DistTentOptimizerHook(TentOptimizerHook):
    def __init__(self, type='online', grad_clip=None, reset=False, repeat=1, coalesce=True, bucket_size_mb=-1):
        super().__init__(type, grad_clip, reset, repeat)
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb