import cv2
import os
import copy
import numpy as np
from PIL import Image

from tools import os_walk

class PersonReIDSamples:

    def __init__(self, dataset_path):

        self.dataset_path = os.path.join(dataset_path, 'bounding_box_train/')
        samples = self._load_samples(self.dataset_path)
        samples = self._reorder_labels(samples, 1)

        self.samples = samples

    def _reorder_labels(self, samples, label_index):

        ids = []
        for sample in samples:
            ids.append(sample[label_index])

        ids = list(set(ids))
        ids.sort()

        for sample in samples:
            sample[label_index] = ids.index(sample[label_index])
        return samples

    def _load_samples(self, floder_dir):
        samples = []
        root_path, _, files_name = os_walk(floder_dir)
        for file_name in files_name:
            if 'jpg' in file_name:
                identity, camera = self._analysis_file_name(file_name)
                samples.append([root_path + file_name, identity, camera])

        return samples

    def _analysis_file_name(self, file_name):
        split_list = file_name.replace('.jpg', '').replace('c', '').split('_')
        identi_id, camera_id = int(split_list[0]), int(split_list[1])
        return identi_id, camera_id

class Samples4OccludedDuke(PersonReIDSamples):

    pass

class Samples4OccludedReID(PersonReIDSamples):

    def _load_samples(self, floder_dir):
        samples = []
        root_path, _, files_name = os_walk(floder_dir)
        for file_name in files_name:
            if 'jpg' in file_name or 'tif' in file_name or 'png' in file_name:
                identity = self._analysis_file_name(file_name)
                samples.append([root_path + file_name, identity, 0])

        return samples

    def _analysis_file_name(self, file_name):
        split_list = file_name.replace('.tif', '').split('_')
        identity_id = int(split_list[0])
        return identity_id

class Samples4PartialDuke(PersonReIDSamples):

    def _load_samples(self, floder_dir):
        samples = []
        root_path, _, files_name = os_walk(floder_dir)
        for file_name in files_name:
            if 'jpg' in file_name or 'tif' in file_name or 'png' in file_name:
                identity = self._analysis_file_name(file_name)
                samples.append([root_path + file_name, identity, 0])

        return samples

    def _analysis_file_name(self, file_name):
        split_list = file_name.replace('.jpg', '').split('_')
        identity_id = int(split_list[0])
        return identity_id

class Samples4Market(PersonReIDSamples):

    def _analysis_file_name(self, file_name):
        split_list = file_name.replace('.jpg', '').replace('c', '').replace('s', '_').split('_')
        identity_id, camera_id = int(split_list[0]), int(split_list[1])
        return identity_id, camera_id

class Samples4Duke(PersonReIDSamples):

    pass

class Samples4MSMT17(PersonReIDSamples):

    def _analysis_file_name(self, file_name):

        split_list = file_name.replace('.jpg', '').replace('c', '').split('_')
        identi_id, camera_id = int(split_list[0]), int(split_list[1])
        return identi_id, camera_id

class TestPersonReIDSamples:

    def __init__(self, dataset_path):

        self.query_path = os.path.join(dataset_path, 'query/')
        self.gallery_path = os.path.join(dataset_path, 'bounding_box_test/')
        query_samples = self._load_samples(self.query_path)
        gallery_samples = self._load_samples(self.gallery_path)
        self.query_samples = query_samples
        self.gallery_samples = gallery_samples

    def _load_samples(self, floder_dir):
        samples = []
        root_path, _, files_name = os_walk(floder_dir)
        for file_name in files_name:
            if 'jpg' in file_name:
                identity_id, camera_id = self._analysis_file_name(file_name)
                samples.append([root_path + file_name, identity_id, camera_id])
        return samples

    def _analysis_file_name(self, file_name):
        split_list = file_name.replace('.jpg', '').replace('c', '').split('_')
        identity_id, camera_id = int(split_list[0]), int(split_list[1])
        return identity_id, camera_id

class TestSamples4OccludedDuke(TestPersonReIDSamples):

    pass

class TestSamples4Market(TestPersonReIDSamples):

    def _analysis_file_name(self, file_name):
        split_list = file_name.replace('.jpg', '').replace('c', '').replace('s', '_').split('_')
        identity_id, camera_id = int(split_list[0]), int(split_list[1])
        return identity_id, camera_id

class TestSamples4Duke(TestPersonReIDSamples):

    pass

class TestSamples4MSMT17(TestPersonReIDSamples):

    def _analysis_file_name(self, file_name):
        split_list = file_name.replace('.jpg', '').replace('c', '').split('_')
        identi_id, camera_id = int(split_list[0]), int(split_list[1])
        return identi_id, camera_id

class TestOccludedReIDSamples:

    def __init__(self, dataset_path):

        self.query_path = os.path.join(dataset_path, 'query/')
        self.gallery_path = os.path.join(dataset_path, 'bounding_box_test/')
        query_samples = self._load_samples(self.query_path, 1)
        gallery_samples = self._load_samples(self.gallery_path, 2)
        self.query_samples = query_samples
        self.gallery_samples = gallery_samples

    def _load_samples(self, floder_dir, camera_id):
        samples = []
        root_path, _, files_name = os_walk(floder_dir)
        for file_name in files_name:
            if 'jpg' in file_name or 'tif' in file_name or 'png' in file_name:
                identity_id = self._analysis_file_name(file_name)
                samples.append([root_path + file_name, identity_id, camera_id])
        return samples

    def _analysis_file_name(self, file_name):
        split_list = file_name.replace('.tif', '').split('_')
        identity_id = int(split_list[0])
        return identity_id

class TestSamples4OccludedReid(TestOccludedReIDSamples):

    pass

class TestSamples4PartialDuke(TestOccludedReIDSamples):

    def _analysis_file_name(self, file_name):
        split_list = file_name.replace('.jpg', '').split('_')
        identity_id = int(split_list[0])
        return identity_id

class Dataset:

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        this_sample = copy.deepcopy(self.samples[index])
        this_sample[0] = self._loader(this_sample[0])
        if self.transform is not None:
            this_sample[0] = self.transform(this_sample[0])
        this_sample[1] = np.array(this_sample[1])
        this_sample[2] = np.array(this_sample[2])
        return this_sample

    def __len__(self):
        return len(self.samples)

    def _loader(self, img_path):
        if 'jpg' in img_path:
            return Image.open(img_path).convert('RGB')
        else:
            return Image.fromarray(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
