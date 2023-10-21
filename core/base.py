import os
import torch
import torch.nn as nn
import torch.optim as optim

from bisect import bisect_right
from network import Model, Res50IBNaBNNeck, Classifier
from tools import CrossEntropyLabelSmooth, os_walk

class Base:
    def __init__(self, config):
        self.config = config

        self.pid_num = config.pid_num

        self.module = config.module
        self.backbone = config.backbone

        self.max_save_model_num = config.max_save_model_num
        self.output_path = config.output_path
        self.save_model_path = os.path.join(self.output_path, 'models/')
        self.save_logs_path = os.path.join(self.output_path, 'logs/')

        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.milestones = config.milestones

        self._init_device()
        self._init_model()
        self._init_creiteron()
        self._init_optimizer()

    def _init_device(self):
        self.device = torch.device('cuda')

    def _init_model(self):

        if self.config.backbone == 'resnet50ibna':
            self.model = Res50IBNaBNNeck()
        else:
            self.model = Model()
        self.model = nn.DataParallel(self.model).to(self.device)

        self.classifier = Classifier(self.pid_num)
        self.classifier = nn.DataParallel(self.classifier).to(self.device)

    def _init_creiteron(self):
        self.pid_creiteron = CrossEntropyLabelSmooth()

    def _init_optimizer(self):

        model_params_group = [{'params': self.model.parameters(), 'lr': self.learning_rate,
                         'weight_decay': self.weight_decay}]
        classifier_params_group = [{'params': self.classifier.parameters(),
                                    'lr': self.learning_rate, 'weight_decay': self.weight_decay}]

        self.model_optimizer = optim.Adam(model_params_group)
        self.model_lr_scheduler = WarmupMultiStepLR(self.model_optimizer, self.milestones,
                                             gamma=0.1, warmup_factor=0.01, warmup_iters=10)

        self.classifier_optimizer = optim.Adam(classifier_params_group)
        self.classifier_lr_scheduler = WarmupMultiStepLR(self.classifier_optimizer, self.milestones,
                                                         gamma=0.1, warmup_factor=0.01, warmup_iters=10)


    def save_model(self, save_epoch, is_best):
        if is_best:
            model_file_path = os.path.join(self.save_model_path, 'model_{}.pth'.format(save_epoch))
            torch.save(self.model.state_dict(), model_file_path)

            classifier_file_path = os.path.join(self.save_model_path, 'classifier_{}.pth'.
                                                              format(save_epoch))
            torch.save(self.classifier.state_dict(), classifier_file_path)


        if self.max_save_model_num > 0:
            root, _, files = os_walk(self.save_model_path)
            for file in files:
                if '.pth' not in file:
                    files.remove(file)
            if len(files) > 2 * self.max_save_model_num:
                file_iters = sorted([int(file.replace('.pth', '').split('_')[1]) for file in files], reverse=False)
                model_file_path = os.path.join(root, 'model_{}.pth'.format(file_iters[0]))
                os.remove(model_file_path)

                classifier_file_path = os.path.join(root, 'classifier_{}.pth'.
                                                                  format(file_iters[0]))
                os.remove(classifier_file_path)


    def resume_last_model(self):
        root, _, files = os_walk(self.save_model_path)
        for file in files:
            if '.pth' not in file:
                files.remove(file)
        if len(files) > 0:
            indexes = []
            for file in files:
                indexes.append(int(file.replace('.pth', '').split('_')[-1]))
            indexes = sorted(list(set(indexes)), reverse=False)
            self.resume_model(indexes[-1])
            start_train_epoch = indexes[-1]
            return start_train_epoch
        else:
            return 0

    def resume_model(self, resume_epoch):
        model_path = os.path.join(self.save_model_path, 'model_{}.pth'.format(resume_epoch))
        self.model.load_state_dict(torch.load(model_path), strict=False)
        print('Successfully resume model from {}'.format(model_path))

        classifier_path = os.path.join(self.save_model_path, 'classifier_{}.pth'.
                                                     format(resume_epoch))
        self.classifier.load_state_dict(torch.load(classifier_path), strict=False)
        print('Successfully resume classifier from {}'.format(classifier_path))

    def set_train(self):
        self.model = self.model.train()
        self.classifier = self.classifier.train()

        self.training = True

    def set_eval(self):
        self.model = self.model.eval()
        self.classifier = self.classifier.eval()

        self.training = False

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=500,
                 warmup_method='linear', last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of " " increasing integers. Got {}", milestones)

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup method accepted got {}".format(warmup_method))
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / float(self.warmup_iters)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
