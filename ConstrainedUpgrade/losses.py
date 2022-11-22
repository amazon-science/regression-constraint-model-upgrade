import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAlignmentLoss(nn.Module):
    
    def __init__(self, margin=3.0, contrastive=False, reduction='none'):
        super(FeatureAlignmentLoss, self).__init__()
        self.margin = margin
        self.contrastive = contrastive
        assert reduction in ['none', 'mean', 'sum']
        self.reduction = reduction

        
    def forward(self, new_feature, target_feature, label):
        if new_feature.size(1) > target_feature.size(1):
            new_feature = new_feature.clone()[:, :target_feature.size(1)]
            new_feature = new_feature.contiguous()
        if self.contrastive:
            pairwise_dist = torch.cdist(new_feature, target_feature, p=2)
            indicator = (label == label[:, None])
            loss = indicator * 0.5 * pairwise_dist ** 2 + (~indicator) * 0.5 * torch.clamp(self.margin - pairwise_dist, min=0) ** 2 
        else:
            loss = F.mse_loss(new_feature, target_feature, reduction='none')
        loss = loss.mean(1)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        else:
            pass
        return loss


class RandomProjectionLoss(nn.Module):
    
    def __init__(self, dim_in_old=512, dim_in_new=512, dim_out=128, project_normalize=False, project_bias=False,
                 project_share=False, project_new_frozen=False, reduction='none'):
        super(RandomProjectionLoss, self).__init__()
        self.project_share = project_share
        assert reduction in ['none', 'mean', 'sum']
        self.reduction = reduction
        self.rp_old = nn.Linear(dim_in_old, dim_out, bias=project_bias)
        nn.init.normal_(self.rp_old.weight, mean=0., std=(1./dim_out)**0.5)
        if not self.project_share:
            self.rp_new = nn.Linear(dim_in_new, dim_out, bias=project_bias)
            nn.init.normal_(self.rp_new.weight, mean=0., std=(1./dim_out)**0.5)
        if project_normalize:
            self.rp_old.weight.data = F.normalize(self.rp_old.weight.data, dim=0)
            if not self.project_share:
                self.rp_new.weight.data = F.normalize(self.rp_new.weight.data, dim=0)
        self.rp_old.weight.requires_grad = False 
        if self.rp_old.bias is not None:
            self.rp_old.bias.requires_grad = False
        if not self.project_share and project_new_frozen:
            self.rp_new.weight.requires_grad = False
            if self.rp_new.bias is not None:
                self.rp_new.bias.requires_grad = False

    def forward(self, new_feature, target_feature):
        project_old = self.rp_old(target_feature)
        if not self.project_share:
            project_new = self.rp_new(new_feature)
        else:
            project_new = self.rp_old(new_feature)
        loss = F.mse_loss(project_new, project_old, reduction=self.reduction)
        return loss


class RetentiveCrossEntropyLoss(nn.Module):

    def __init__(self, two_heads=False, reduction='none'):
        super(RetentiveCrossEntropyLoss, self).__init__()
        self.two_heads = two_heads
        assert reduction in ['none', 'mean', 'sum']
        self.reduction = reduction

    def forward(self, new_logits, target_logits, label):
        num_classes = new_logits.size(1)
        combined_logits = torch.empty_like(new_logits)
        for i in range(new_logits.size(0)):
            combined_logits[i, :] = target_logits[i, :]
            combined_logits[i, label[i]] = new_logits[i, label[i]]
        loss = F.cross_entropy(combined_logits, label, reduction=self.reduction)
        if self.two_heads:
            combined_logits_2 = torch.empty_like(new_logits)
            for i in range(new_logits.size(0)):
                combined_logits_2[i, :] = new_logits[i, :]
                combined_logits_2[i, label[i]] = target_logits[i, label[i]]
            loss += F.cross_entropy(combined_logits_2, label, reduction=self.reduction)
        return loss


class LogitsInhibitionLoss(nn.Module):
    
    def __init__(self, p=2, margin=0., one_sided=False, exclude_gt=False, compute_topk=-1,
                 enhance_gt=False, enhance_gt_weight=0.0, is_margin_relative=False,
                 use_p_norm=False, reduction='none'):
        super(LogitsInhibitionLoss, self).__init__()
        self.p = p
        self.margin = margin
        self.one_sided = one_sided
        assert not (exclude_gt & enhance_gt)
        self.exclude_gt = exclude_gt
        self.compute_topk = compute_topk
        self.enhance_gt = enhance_gt
        self.enhance_gt_weight = enhance_gt_weight
        self.is_margin_relative = is_margin_relative
        self.use_p_norm = use_p_norm
        if p > 2 and not use_p_norm:
            print('using p > 2 may cause numerical issues without using norm.')
        assert reduction in ['none', 'mean', 'sum']
        self.reduction = reduction

    def forward(self, new_logits, target_logits, label):
        if len(label.shape) > 1:  # for swin-transformer label smoothing
            label = label.argmax(dim=1)
        if self.one_sided:
            diff_logits = new_logits - target_logits
            diff_logits = torch.clamp(diff_logits - self.margin, min=0)
            y_hot = F.one_hot(label, num_classes=new_logits.size(1))
            diff2_logits = target_logits - new_logits
            diff2_logits = torch.clamp(diff2_logits - self.margin, min=0)
            if self.exclude_gt:
                loss = (torch.ones_like(y_hot) - y_hot) * (diff_logits) ** (self.p)
            elif self.enhance_gt:
                loss = (torch.ones_like(y_hot) - y_hot) * (diff_logits) ** (self.p) + self.enhance_gt_weight * y_hot * (diff2_logits) ** (self.p)
            else:
                loss = (diff_logits) ** (self.p)
        else:
            if self.is_margin_relative:
                target_logits_maxima = target_logits.max(1)[0]
                diff_logits = torch.abs(new_logits - target_logits) / target_logits_maxima[:, None]
            else:
                diff_logits = torch.abs(new_logits - target_logits)
            diff_logits = torch.clamp(diff_logits - self.margin, min=1e-7)
            y_hot = F.one_hot(label, num_classes=new_logits.size(1))
            if self.compute_topk > 0:
                indices = new_logits.argsort(descending=True)[:self.compute_topk]
                if self.exclude_gt:
                    loss = ((torch.ones_like(y_hot) - y_hot))[:, indices] * (diff_logits[:, indices]) ** (self.p)
                else:
                    loss = (diff_logits[:, indices]) ** (self.p)
            elif self.exclude_gt:
                loss = (torch.ones_like(y_hot) - y_hot) * (diff_logits) ** (self.p)
            else:
                loss = (diff_logits) ** (self.p)
        # for large p (>2), use norm instead of power for numerical issues
        if self.use_p_norm:
            loss = loss.sum(1) ** (1. / self.p)
        else:
            loss = loss.mean(1)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        else:
            pass
        return loss


class RenyiDivLoss(nn.Module):

    def __init__(self, reduction=False, alpha=2, reorder=False, margin=0.):
        super(RenyiDivLoss, self).__init__()
        self.reduction = reduction
        assert (alpha > 0 and alpha != 1)
        self.alpha = alpha
        self.reorder = reorder
        self.margin = margin 

    
    def forward(self, new_prob, target_prob):
        if self.reorder:
            new_prob_ = torch.sort(new_prob, dim=1)[0]
            target_prob_ = torch.sort(target_prob, dim=1)[0]
        else:
            new_prob_ = new_prob.clone()
            target_prob_ = target_prob.clone()
        new_prob_ = torch.clamp(new_prob_, min=torch.finfo(new_prob_.dtype).eps)
        target_prob_ = torch.clamp(target_prob_, min=torch.finfo(target_prob_.dtype).eps)

        r = (target_prob_ / new_prob_) ** (self.alpha - 1) * target_prob_
        renyi = torch.log(r.sum(1)) / (self.alpha - 1)
        if self.reduction:
            return renyi.mean(0) 
        else:
            return renyi


class CNALoss(nn.Module):
    """
    Sample-wise CNA distilation loss
    Can be either
    - supervised: uses groundtruth labels
    or
    - unsupervised: does not need groundtruth labels
    """

    def __init__(self, reduction=False, supervised=False,
                 topk=1, temperature=0.01, summation_order='out'):
        super(CNALoss, self).__init__()

        self.supervised = supervised
        self.topk = topk
        self.tau = temperature
        self.sum_order = summation_order
        self.reduction = reduction
        self.temperature = temperature

    def forward(self, student_x, teacher_x, y):
        """
        student_x: Nxd
        teacher_x: Nxd
        y: d
        """

        # normalize features
        norm_student_x = F.normalize(student_x, dim=1)
        norm_teacher_x_d = F.normalize(teacher_x.detach(), dim=1) # detach teacher output to avoid parameter update

        # get ordering
        teacher_innder_product = torch.mm(norm_teacher_x_d, norm_teacher_x_d.T)

        # get sorting mask
        eye_mask = 1 - torch.eye(teacher_innder_product.shape[0],
                             device=teacher_innder_product.device,
                             dtype=teacher_innder_product.dtype) # self is excluded

        if self.supervised:
            # mask out samples from different classes if using supervision
            same_class = torch.eq(y[:, None], y[:, None])
            mask = eye_mask * same_class
        else:
            mask = eye_mask

        # normalized distance are in the range of [-1, 1], we can add a constant to them without disrupting the order
        masksed_teacher_inner_product = (teacher_innder_product + 2) * mask

        closest_idx = masksed_teacher_inner_product.argsort(dim=1, descending=True)
        topk_idx = closest_idx[:, :self.topk]

        is_positive = torch.eq(topk_idx[:, :, None],
                               torch.arange(0, teacher_x.shape[0],
                                            dtype=topk_idx.dtype, device=topk_idx.device)[None, None, :])

        loss_mask = torch.detach(is_positive.sum(dim=1) * mask) # gives NxN 0-1 mask

        # calcuate the loss given the teacher information
        scaled_student_inner_product = torch.mm(norm_student_x, norm_student_x.T) / self.temperature

        # suppress the potential overflow
        supp_scaled_student_ip = scaled_student_inner_product - scaled_student_inner_product.max(dim=1, keepdim=True)[0].detach()

        exp_ss_student_ip = torch.exp(supp_scaled_student_ip)
        log_denorm = torch.log((exp_ss_student_ip * eye_mask).sum(dim=1))

        if self.sum_order == 'in':
            # loss i = - log( \sum_{j \in P(i)} exp(<x_i, x_j> / t) / \sum_{j \ A(i)} exp(<x_i, x_j> / t))
            summation = (exp_ss_student_ip * loss_mask).sum(dim=1) + 1e-15
            loss = torch.where(torch.gt(loss_mask.sum(dim=1), 0), log_denorm - torch.log(summation), torch.zeros_like(log_denorm))
        elif self.sum_order == 'out':
            # loss
            loss = ((log_denorm[:, None] - supp_scaled_student_ip) * loss_mask).sum(dim=1) / (loss_mask.sum(dim=1).detach() + 1e-15)
        else:
            raise ValueError("Unknown summation order: {}".format(self.sum_order))

        if self.reduction:
            return loss.mean()
        else:
            return loss


class FocalDistillationLoss(nn.Module):
    """
    This class implements the focal distillation loss for PC-training.

    It requires three input data:
    - New model's prediction
    - Old model's prediction
    - groundtruth

    It supports three types of Distillation loss
    - Focal distillation with the KL-divergence loss (default)
    - Focal distillation with L2 loss
    - Focal distillation with the CNA loss (requires feature before SoftMax as input)

    There are two parameters adjusting the focal weights
    - fd_alpha: the weight of the background samples (default = 1)
    - fd_beta: the weight of the focused samples (default = 0)

    fd_alpha = 1 and fd_beta = 0 resembles the normal knowledge distillation setup

    This loss function is meant to be used as a additional loss term to a classification loss function (e.g. CrossEntropy)

    """

    def __init__(self, fd_alpha=1, fd_beta=0,
                 focus_type='old_correct',
                 distillation_type='kl',
                 kl_temperature=5,
                 cna_temperature=0.05,
                 renyi_alpha=2,
                 renyi_reorder=False,
                 renyi_margin=0.,
                 li_p=2,
                 li_margin=1.0,
                 li_one_sided=False,
                 li_exclude_gt=False,
                 li_compute_topk=-1,
                 li_enhance_gt=False,
                 li_enhance_gt_weight=0.0,
                 li_margin_relative=False,
                 li_use_p_norm=False,
                 rce_two_heads=False,
                 rp_dim_in_old=512,
                 rp_dim_in_new=512,
                 rp_dim_out=128,
                 rp_use_bias=False,
                 rp_project_normalize=False,
                 rp_project_share=False,
                 rp_project_new_frozen=False,
                 fa_margin=3.0,
                 fa_contrastive=False,
                 li_cna_alpha=1.0,
                 ):
        super(FocalDistillationLoss, self).__init__()

        self._fd_alpha = fd_alpha
        self._fd_beta = fd_beta

        self._fd_type = focus_type
        self._distill_type = distillation_type

        if distillation_type == 'kl':
            self.distill_loss = nn.KLDivLoss(reduction='none')
        elif distillation_type == 'l2':
            self.distill_loss = nn.MSELoss(reduction='none')
        elif distillation_type == 'cna':
            self.distill_loss = CNALoss(temperature=cna_temperature)
        elif distillation_type == 'renyi':
            self.distill_loss = RenyiDivLoss(alpha=renyi_alpha, reorder=renyi_reorder, margin=renyi_margin)
        elif distillation_type == 'li':
            self.distill_loss = LogitsInhibitionLoss(p=li_p, margin=li_margin, one_sided=li_one_sided,
                                                      exclude_gt=li_exclude_gt, compute_topk=li_compute_topk,
                                                      enhance_gt=li_enhance_gt, enhance_gt_weight=li_enhance_gt_weight,
                                                      is_margin_relative=li_margin_relative,
                                                      use_p_norm=li_use_p_norm, reduction='none')
        elif distillation_type == 'rce':
            self.distill_loss = RetentiveCrossEntropyLoss(two_heads=rce_two_heads, reduction='none')
        elif distillation_type == 'rp':
            self.distill_loss = RandomProjectionLoss(dim_in_old=rp_dim_in_old, dim_in_new=rp_dim_in_new, dim_out=rp_dim_out,
                                                     project_normalize=rp_project_normalize, project_bias=rp_use_bias,
                                                     project_share=rp_project_share,
                                                     project_new_frozen=rp_project_new_frozen, reduction='none')
        elif distillation_type == 'fa':
            self.distill_loss = FeatureAlignmentLoss(margin=fa_margin, contrastive=fa_contrastive, reduction='none')
        elif distillation_type == 'li+cna':
            self.distill_loss_1 = LogitsInhibitionLoss(p=li_p, margin=li_margin, one_sided=li_one_sided,
                                                      exclude_gt=li_exclude_gt, compute_topk=li_compute_topk,
                                                      enhance_gt=li_enhance_gt, enhance_gt_weight=li_enhance_gt_weight,
                                                      is_margin_relative=li_margin_relative,
                                                      use_p_norm=li_use_p_norm, reduction='none')
            self.distill_loss_2 = CNALoss(temperature=cna_temperature)
            self.li_cna_alpha = li_cna_alpha
        else:
            raise ValueError("Unknown loss type: {}".format(self._distill_type))

        self._kl_temperature = kl_temperature

    def forward(self,
                new_model_prediction, old_model_prediction,
                gt,
                new_model_feature, old_model_feature,
                kd_likelihood=None):

        if type(new_model_prediction).__name__ == 'BCTOutputs':
            new_model_prediction, bct_output = new_model_prediction
        old_cls_num = old_model_prediction.size(1)
        new_cls_num = new_model_prediction.size(1)
        if old_cls_num != new_cls_num:
            #TODO: may generate empty tensor, need to be fixed
            mask = gt < min(new_cls_num, old_cls_num) # mask samples belong to new classes
            gt = gt[mask]
            new_model_prediction = new_model_prediction[mask, :min(new_cls_num, old_cls_num)] # align output dim
            old_model_prediction = old_model_prediction[mask, :min(new_cls_num, old_cls_num)]

        if self._fd_type != 'all_pass':
            # get old model prediction
            old_model_correct = old_model_prediction.argmax(dim=1) == gt
            new_model_correct = new_model_prediction.argmax(dim=1) == gt

        if self._fd_type == 'old_correct':
            loss_weights = old_model_correct.int()[:, None] * self._fd_beta + self._fd_alpha
        elif self._fd_type == 'neg_flip':
            loss_weights = (old_model_correct & (~new_model_correct)).int().unsqueeze(1) * self._fd_beta + self._fd_alpha
        elif self._fd_type == 'new_incorrect':
            loss_weights = (~new_model_correct).int().unsqueeze(1) * self._fd_beta + self._fd_alpha
        elif self._fd_type == 'all_pass':
            loss_weights = 1
        elif self._fd_type == 'likelihood':
            loss_weights = kd_likelihood
        else:
            raise ValueError("Unknown focus type: {}".format(self._fd_type))

        # get per-sample loss
        if self._distill_type == 'kl':
            sample_loss = self.distill_loss(
                F.log_softmax(new_model_prediction / self._kl_temperature, dim=1),
                F.softmax(old_model_prediction / self._kl_temperature, dim=1)).sum(dim=1) * (self._kl_temperature ** 2)
        elif self._distill_type == 'l2':
            sample_loss = self.distill_loss(new_model_prediction, old_model_prediction)
        elif self._distill_type == 'cna':
            sample_loss = self.distill_loss(new_model_feature, old_model_feature, gt)
        elif self._distill_type == 'renyi':
            sample_loss = self.distill_loss(
                F.softmax(new_model_prediction, dim=1),
                F.softmax(old_model_prediction, dim=1))
            # sample_loss = self.distill_loss(
            #     F.softmax(old_model_prediction, dim=1),
            #     F.softmax(new_model_prediction, dim=1))
        elif self._distill_type == 'li':
            sample_loss = self.distill_loss(new_model_prediction, old_model_prediction, gt)
        elif self._distill_type == 'rce':
            sample_loss = self.distill_loss(new_model_prediction, old_model_prediction, gt)
        elif self._distill_type == 'rp':
            sample_loss = self.distill_loss(new_model_feature, old_model_feature)
        elif self._distill_type == 'fa':
            sample_loss = self.distill_loss(new_model_feature, old_model_feature, gt)
        elif self._distill_type == 'li+cna':
            sample_loss = self.li_cna_alpha * self.distill_loss_1(new_model_prediction, old_model_prediction, gt) + (1 - self.li_cna_alpha) * self.distill_loss_2(new_model_feature, old_model_feature, gt)
        else:
            raise ValueError("Unknown loss type: {}".format(self._distill_type))

        # weighted sum of losses
        return (sample_loss * loss_weights).mean()


if __name__ == '__main__':
    N = 5
    d = 100
    c = 1028

    torch.manual_seed(112)

    new_pred = torch.rand(N, d, dtype=torch.float32).cuda()
    new_pred.requires_grad_(True)
    old_pred = torch.rand(N, d, dtype=torch.float32).cuda()

    new_feat = torch.rand(N, c, dtype=torch.float32).cuda()
    new_feat.requires_grad_(True)
    old_feat = torch.rand(N, c, dtype=torch.float32).cuda()

    y = torch.randint(0, d, (N, )).cuda()

    m = FocalDistillationLoss(kl_temperature=100).cuda()

    loss = m(new_pred, old_pred, y, new_feat, old_feat)

    loss.backward()

    print(loss)

    m_l2 = FocalDistillationLoss(distillation_type='l2')

    loss2 = m_l2(new_pred, old_pred, y, new_feat, old_feat)

    loss2.backward()

    print(loss2)

    m_cna = FocalDistillationLoss(distillation_type='cna', cna_temperature=0.01)

    loss3 = m_cna(new_pred, old_pred, y, new_feat, old_feat)

    # loss3.backward()

    print(loss3)

    m_ref = LSALoss(0.0, 1.0, 0.01, 0, 1, 1, 1, 1)

    loss_ref = m_ref(new_pred, y, new_feat, old_feat)

    print(loss_ref)


