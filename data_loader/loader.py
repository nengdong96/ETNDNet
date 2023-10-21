
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data_loader.preprocessing import RandomErasing
from data_loader.dataset import Samples4OccludedDuke, Samples4OccludedReID,  Samples4PartialDuke, Samples4Market, \
    Samples4Duke,  Samples4MSMT17, TestSamples4OccludedDuke, TestSamples4OccludedReid, TestSamples4PartialDuke,\
    TestSamples4Market, TestSamples4Duke,  TestSamples4MSMT17, Dataset
from data_loader.sampler import TripletSampler

class Loader:

    def __init__(self, config):
        transform_train = [
            transforms.Resize(config.image_size, interpolation=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop(config.image_size)]
        if config.use_colorjitor:
            transform_train.append(transforms.ColorJitter(brightness=0.25, contrast=0.15, saturation=0.25, hue=0))
        transform_train.extend([transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        if config.use_rea:
            transform_train.append(RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]))
        self.transform_train = transforms.Compose(transform_train)

        self.transform_test = transforms.Compose([
            transforms.Resize(config.image_size, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.datasets = ['occluded_duke, occluded_reid, partial_reid, market, duke']
        self.train_dataset = config.train_dataset
        self.test_dataset = config.test_dataset

        self.occluded_duke_path = config.occluded_duke_path
        self.occluded_reid_path = config.occluded_reid_path
        self.partial_duke_path = config.partial_duke_path
        self.market_path = config.market_path
        self.duke_path = config.duke_path
        self.msmt_path = config.msmt_path
        self.batchsize = config.batchsize
        self.num_instances = config.num_instances

        self._load()

    def _load(self):
        samples = self._get_samples(self.train_dataset)
        self.loader = self._get_train_iter(samples, self.transform_train, self.batchsize)
        query_samples, gallery_samples = self._get_test_samples(self.test_dataset)
        self.query_loader = self._get_test_loader(query_samples, self.transform_test, 128)
        self.gallery_loader = self._get_test_loader(gallery_samples, self.transform_test, 128)

    def _get_samples(self, dataset):
        if dataset == 'occluded_duke':
            samples = Samples4OccludedDuke(self.occluded_duke_path)
        elif dataset == 'occluded_reid':
            samples = Samples4OccludedReID(self.occluded_reid_path)
        elif dataset == 'partial_duke':
            samples = Samples4PartialDuke(self.partial_duke_path)
        elif dataset == 'market':
            samples = Samples4Market(self.market_path)
        elif dataset == 'duke':
            samples = Samples4Duke(self.duke_path)
        elif dataset == 'msmt':
            samples = Samples4MSMT17(self.msmt_path)
        return samples

    def _get_test_samples(self, dataset):
        if dataset == 'occluded_duke':
            query_samples = TestSamples4OccludedDuke(self.occluded_duke_path).query_samples
            gallery_samples = TestSamples4OccludedDuke(self.occluded_duke_path).gallery_samples
        elif dataset == 'occluded_reid':
            query_samples = TestSamples4OccludedReid(self.occluded_reid_path).query_samples
            gallery_samples = TestSamples4OccludedReid(self.occluded_reid_path).gallery_samples
        elif dataset == 'partial_duke':
            query_samples = TestSamples4PartialDuke(self.partial_duke_path).query_samples
            gallery_samples = TestSamples4PartialDuke(self.partial_duke_path).gallery_samples
        elif dataset == 'market':
            query_samples = TestSamples4Market(self.market_path).query_samples
            gallery_samples = TestSamples4Market(self.market_path).gallery_samples
        elif dataset == 'duke':
            query_samples = TestSamples4Duke(self.duke_path).query_samples
            gallery_samples = TestSamples4Duke(self.duke_path).gallery_samples
        elif dataset == 'msmt':
            query_samples = TestSamples4MSMT17(self.msmt_path).query_samples
            gallery_samples = TestSamples4MSMT17(self.msmt_path).gallery_samples

        return query_samples, gallery_samples

    def _get_train_iter(self, samples, transform, batchsize):
        dataset = Dataset(samples.samples, transform=transform)
        loader = DataLoader(dataset, batch_size=batchsize, sampler=TripletSampler(dataset.samples, batchsize,
                            self.num_instances), num_workers=8)
        return loader

    def _get_test_loader(self, samples, transform, batch_size):

        dataset = Dataset(samples, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=False)
        return loader

class IterLoader:

    def __init__(self, loader):
        self.loader = loader
        self.iter = iter(self.loader)

    def next_one(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)




