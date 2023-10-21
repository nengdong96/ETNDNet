import torch.nn as nn
from .metric import *


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = targets.long()
        size = log_probs.size()
        targets = torch.zeros((size[0], size[1])).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)

        if self.use_gpu:
            targets = targets.to(torch.device('cuda'))
        targets = (1 - self.epsilon) * targets + self.epsilon / size[1]
        loss = (-targets * log_probs).mean(0).sum()
        return loss

class MixingCrossEntropyLabelSmooth(nn.Module):
    def __init__(self, theta, epsilon=0.1, use_gpu=True):
        super(MixingCrossEntropyLabelSmooth, self).__init__()
        self.theta = theta
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, mixing_targets):
        log_probs = self.logsoftmax(inputs)
        targets = targets.long()
        size = log_probs.size()
        targets = torch.zeros((size[0], size[1])).scatter_(1, targets.unsqueeze(1).data.cpu(), 1) * self.theta
        mixing_targets = torch.zeros((size[0], size[1])).scatter_(1, mixing_targets.unsqueeze(1).data.cpu(), 1) * (1.0 - self.theta)
        new_targets = targets + mixing_targets
        if self.use_gpu:
            new_targets = new_targets.to(torch.device('cuda'))
        new_targets = (1 - self.epsilon) * new_targets + self.epsilon / size[1]
        loss = (-new_targets * log_probs).mean(0).sum()
        return loss

def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def cosine_dist(x, y):
    dist = 1 - torch.matmul(x, y.t())
    return dist

def euclidean_dist(x, y):

    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def euclidean_dist_elementwise(x, y):
    xx = torch.pow(x, 2).sum(1)
    yy = torch.pow(y, 2).sum(1)
    xy = (x * y).sum(1)
    dist = xx + yy - 2 * xy
    dist = dist.clamp(min=1e-12).sqrt()
    return dist

def hard_example_mining(dist_mat, labels, kthp=1, kthn=1, return_inds=False):

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    dist_ap = torch.max(dist_mat * is_pos.float().detach(), 1, keepdim=True)[0]
    dist_an = torch.min(torch.max(is_pos.float() * 1000, dist_mat * is_neg.float().detach()), 1, keepdim=True)[0]

    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    return dist_ap, dist_an

class TripletLoss(object):

    def __init__(self, margin=None, kthp=1, kthn=1):
        self.margin = margin
        self.kthp = kthp
        self.kthn = kthn
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction="mean")
        else:
            self.ranking_loss = nn.SoftMarginLoss(reduction="mean")

    def __call__(self, global_feat, labels, normalize_feature=True):
        normalize_feature = False
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)

        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels, kthp=self.kthp, kthn=self.kthn)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)

        return loss

def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

















