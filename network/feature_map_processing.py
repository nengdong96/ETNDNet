import math
import random
import torch

class FeatureMapErasing():
    def __init__(self, config):
        super(FeatureMapErasing, self).__init__()
        self.config = config

    def __call__(self, features_map):
        size = features_map.size(0)
        h, w = features_map.size(2), features_map.size(3)

        area = h * w

        erasing_features_map = torch.zeros(features_map.size()).cuda()
        ones_map = torch.ones([h, w]).cuda()

        for attempt in range(100):
            target_e_area = random.uniform(self.config.lower, self.config.upper) * area
            aspect_e_ratio = random.uniform(self.config.ratio, 1 / self.config.ratio)
            target_e_h = int(round(math.sqrt(target_e_area * aspect_e_ratio)))
            target_e_w = int(round(math.sqrt(target_e_area / aspect_e_ratio)))
            if target_e_h < h and target_e_w < w:
                x_e = random.randint(0, w - target_e_w)
                y_e = random.randint(0, h - target_e_h)
                ones_map[y_e: y_e + target_e_h, x_e: x_e + target_e_w] = 0.0
                for i in range(size):
                    erasing_features_map[i] = (features_map[i] * ones_map).unsqueeze(0)
                break

        return erasing_features_map

class FeatureMapTransforming():
    def __init__(self, config):
        super(FeatureMapTransforming, self).__init__()
        self.config = config

    def __call__(self, features_map):
        size = features_map.size(0)
        h, w = features_map.size(2), features_map.size(3)

        area = h * w

        transforming_features_map = features_map.clone()

        for attempt in range(100):
            target_t_area = random.uniform(self.config.lower, self.config.upper) * area
            aspect_t_ratio = random.uniform(self.config.ratio, 1 / self.config.ratio)
            target_t_h = int(round(math.sqrt(target_t_area * aspect_t_ratio)))
            target_t_w = int(round(math.sqrt(target_t_area / aspect_t_ratio)))
            if target_t_h < h and target_t_w < w:
                x_t_1 = random.randint(0, w - target_t_w)
                y_t_1 = random.randint(0, h - target_t_h)
                x_t_2 = random.randint(0, w - target_t_w)
                y_t_2 = random.randint(0, h - target_t_h)
                for i in range(size):
                    transforming_features_map[i][:, y_t_1: y_t_1 + target_t_h, x_t_1: x_t_1 + target_t_w] = \
                        features_map[i][:, y_t_2: y_t_2 + target_t_h, x_t_2: x_t_2 + target_t_w]
                break

        return transforming_features_map

class FeatureMapNoising():
    def __init__(self, config):
        super(FeatureMapNoising, self).__init__()
        self.config = config

    def __call__(self, features_map):
        size = features_map.size(0)
        h, w = features_map.size(2), features_map.size(3)

        area = h * w

        noising_features_map = features_map.clone()
        ones_map = torch.ones(([h, w])).cuda()

        for attempt in range(100):
            target_n_area = random.uniform(self.config.lower, self.config.upper) * area
            aspect_n_ratio = random.uniform(self.config.ratio, 1 / self.config.ratio)
            target_n_h = int(round(math.sqrt(target_n_area * aspect_n_ratio)))
            target_n_w = int(round(math.sqrt(target_n_area / aspect_n_ratio)))
            if target_n_h < h and target_n_w < w:
                x_m = random.randint(0, w - target_n_w)
                y_n = random.randint(0, h - target_n_h)
                noise = torch.rand(size=(target_n_h, target_n_w))
                ones_map[y_n: y_n + target_n_h, x_m: x_m + target_n_w] = noise
                for i in range(size):
                    noising_features_map[i] = (features_map[i] * ones_map).unsqueeze(0)
                break

        return noising_features_map