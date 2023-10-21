
import numpy as np
from sklearn import metrics as sk_metrics

class ReIDEvaluator:

    def __init__(self, dist, mode):
        assert dist in ['cosine', 'euclidean', 'gaussian']
        self.dist = dist
        assert mode in ['inter-camera', 'intra-camera', 'all']
        self.mode = mode

    def evaluate(self, query_features, query_pids, query_cids, gallery_features, gallery_pids, gallery_cids):

        if self.dist is 'cosine':
            scores = self.cosine_dist(query_features, gallery_features)
            rank_results = np.argsort(scores)[:, ::-1]
        elif self.dist is 'euclidean':
            scores = self.euclidean_dist(query_features, gallery_features)
            rank_results = np.argsort(scores)
        elif self.dist is 'gaussian':
            scores = self.gaussian_kernel_function(query_features, gallery_features)
            rank_results = np.argsort(scores)[:, ::-1]

        APs, CMC = [], []
        for idx, data in enumerate(zip(rank_results, query_pids, query_cids)):
            a_rank, query_pid, query_cid = data
            ap, cmc = self.compute_AP(a_rank, query_pid, query_cid, gallery_pids, gallery_cids)
            APs.append(ap), CMC.append(cmc)

        MAP = np.array(APs).mean()
        min_len = min([len(cmc) for cmc in CMC])
        CMC = [cmc[:min_len] for cmc in CMC]
        CMC = np.mean(np.array(CMC), axis=0)

        return MAP, CMC

    def compute_AP(self, a_rank, query_pid, query_cid, gallery_pids, gallery_cids):

        if self.mode == 'inter-camera':
            junk_index_1 = self.in1d(np.argwhere(query_pid == gallery_pids), np.argwhere(query_cid == gallery_cids))
            junk_index_2 = np.argwhere(gallery_pids == -1)
            junk_index = np.append(junk_index_1, junk_index_2)
            index_wo_junk = self.notin1d(a_rank, junk_index)
            good_index = self.in1d(np.argwhere(query_pid == gallery_pids), np.argwhere(query_cid != gallery_cids))

        elif self.mode == 'intra-camera':
            junk_index_1 = np.argwhere(query_cid != gallery_cids)
            junk_index_2 = np.argwhere(gallery_pids == -1)
            junk_index = np.append(junk_index_1, junk_index_2)
            index_wo_junk = self.notin1d(a_rank, junk_index)
            good_index = np.argwhere(query_pid == gallery_pids)
            self_junk = a_rank[0]
            index_wo_junk = np.delete(index_wo_junk, np.where(self_junk == index_wo_junk))
            good_index = np.delete(good_index, np.where(self_junk == good_index))
        elif self.mode == 'all':
            junk_index = np.argwhere(gallery_pids == -1)
            index_wo_junk = self.notin1d(a_rank, junk_index)
            good_index = np.argwhere(query_pid == gallery_pids)
            self_junk = a_rank[0]
            index_wo_junk = np.delete(index_wo_junk, np.where(self_junk == index_wo_junk))
            good_index = np.delete(good_index, np.where(self_junk == good_index))

        hit = np.in1d(index_wo_junk, good_index)
        index_hit = np.argwhere(hit == True).flatten()
        if len(index_hit) == 0:
            AP = 0
            cmc = np.zeros([len(index_wo_junk)])
        else:
            precision = []
            for i in range(len(index_hit)):
                precision.append(float(i + 1) / float((index_hit[i] + 1)))
            AP = np.mean(np.array(precision))
            cmc = np.zeros([len(index_wo_junk)])
            cmc[index_hit[0]:] = 1
        return AP, cmc

    def in1d(self, array1, array2, invert=False):
        mask = np.in1d(array1, array2, invert=invert)
        return array1[mask]

    def notin1d(self, array1, array2):
        return self.in1d(array1, array2, invert=True)

    def cosine_dist(self, x, y):
        def normalize(x):
            norm = np.tile(np.sqrt(np.sum(np.square(x), axis=1, keepdims=True)), [1, x.shape[1]])
            return x / norm
        x = normalize(x)
        y = normalize(y)
        return np.matmul(x, y.transpose([1, 0]))

    def euclidean_dist(self, x, y):
        return sk_metrics.pairwise.euclidean_distances(x, y)

    def gaussian_kernel_function(self, x, y):
        similarity = np.exp(np.square((x - y)))
        return similarity

