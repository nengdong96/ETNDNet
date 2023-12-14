#Need to simply modify the classifier in the model, including 'self.BN(features.squeeze().unsqueeze(0))'
# and 'return bn_features, cls_score'
import os
import ast
import argparse
import random
from PIL import Image
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from core import Base
from tools import os_walk


def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def draw_CAMs(base, img_path, transform):
    img = Image.open(img_path).convert('RGB')
    if transform:
        img = transform(img)
    img = img.unsqueeze(0)
    base.set_eval()
    features_map = base.model(img)
    bn_features, pred = base.classifier(features_map)
    pred_pid = torch.argmax(pred).item()
    size_upsample = (64, 128)
    bz, nc, h, w = features_map.size()
    classifier_name = []
    classifier_params = []
    for name, param in base.classifier.named_parameters():
        classifier_name.append(name)
        classifier_params.append(param)
    heatmap = torch.matmul(torch.ones(classifier_params[-1][pred_pid].unsqueeze(0).size()).cuda(),
                                features_map.unsqueeze(0).reshape(nc, h * w)).detach()
    heatmap = heatmap.reshape(h, w)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = cv2.resize(heatmap.cpu().numpy(), size_upsample)
    heatmap = np.uint8(255 * heatmap)
    heatmap = heatmap.reshape(128, 64, 1)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, (64, 128))
    superimposed_img = (heatmap / 255.0) * original_img
    return superimposed_img
'''
def draw_CAMs(base, img_path, transform):
    img = Image.open(img_path).convert('RGB')
    if transform:
        img = transform(img)
    img = img.unsqueeze(0)
    base.set_eval()
    features_map = base.model(img)
    bn_features, pred = base.classifier(features_map)
    pred_pid = torch.argmax(pred).item()
    size_upsample = (64, 128)
    bz, nc, h, w = features_map.size()
    classifier_name = []
    classifier_params = []
    for name, param in base.classifier.named_parameters():
        classifier_name.append(name)
        classifier_params.append(param)
    heatmap = torch.matmul(classifier_params[-1][pred_pid].unsqueeze(0),
                                features_map.unsqueeze(0).reshape(nc, h * w)).detach()
    heatmap = heatmap.reshape(h, w)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = cv2.resize(heatmap.cpu().numpy(), size_upsample)
    heatmap = np.uint8(255 * heatmap)
    heatmap = heatmap.reshape(128, 64, 1)
    heatmap1 = heatmap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, (64, 128))
    superimposed_img = (heatmap / 255.0) * heatmap1
    return superimposed_img
    '''

def main(config):
    model = Base(config)
    if config.auto_resume_training_from_lastest_step:
        root, _, files = os_walk(model.save_model_path)
        if len(files) > 0:
            indexes = []
            for file in files:
                indexes.append(int(file.replace('.pth', '').split('_')[-1]))
            indexes = sorted(list(set(indexes)), reverse=False)
            model.resume_model(indexes[-1])

    img_size = [256, 128]
    transform = [
        transforms.Resize(img_size, interpolation=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(10),
        transforms.RandomCrop(img_size)]
    transform.extend([transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform = transforms.Compose(transform)

    files = os.listdir('aaa')
    save_path = 'aaa1'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for file in files:
        img = os.path.join('aaa/', file)
        save_name = 'cam_'+file+'.png'
        save_path_name = save_path+'/'+save_name
        cam_img = draw_CAMs(model, img, transform)
        cv2.imwrite(save_path_name, cam_img)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default='train', help='train, test')
    parser.add_argument('--module', type=str, default='B',
                        help='B, ED, TD, ND, ETD, END, TND, ETND')
    parser.add_argument('--backbone', type=str, default='resnet50', help='resnet50, resnet50ibna')
    parser.add_argument('--occluded_duke_path', type=str, default='G:/datasets/Occluded_Duke')
    parser.add_argument('--occluded_reid_path', type=str, default='/opt/data/private/data/Occluded_REID_OURS/new')
    parser.add_argument('--partial_duke_path', type=str, default='/opt/data/private/data/P_Duke_OURS/new')
    parser.add_argument('--market_path', type=str, default='/opt/data/private/data//Market-1501-v15.09.15')
    parser.add_argument('--duke_path', type=str, default='/opt/data/private/data/DukeMTMC-reID')
    parser.add_argument('--msmt_path', type=str, default='/opt/data/private/data/MSMT17')
    parser.add_argument('--train_dataset', type=str, default='occluded_duke', help='occluded_duke, occluded_reid, '
                         'partial_duke, market, duke, msmt')
    parser.add_argument('--test_dataset', type=str, default='occluded_duke', help='occluded_duke, occluded_reid, '
                        'partial_duke, market, duke, msmt')
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 128])
    parser.add_argument('--use_rea', type=ast.literal_eval, default=True, help='use random erasing augmentation')
    parser.add_argument('--use_colorjitor', type=ast.literal_eval, default=False, help='use random erasing augmentation')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--num_instances', type=int, default=8)
    parser.add_argument('--pid_num', type=int, default=702)
    parser.add_argument('--in_dim', type=int, default=2048)
    parser.add_argument('--learning_rate', type=float, default=0.0003)
    parser.add_argument('--lower', type=float, default=0.02)
    parser.add_argument('--upper', type=float, default=0.4)
    parser.add_argument('--ratio', type=float, default=0.3)
    parser.add_argument('--lambda1', type=float, default=0.1)
    parser.add_argument('--lambda2', type=float, default=0.15)
    parser.add_argument('--lambda3', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70],
                        help='milestones for the learning rate decay')
    parser.add_argument('--output_path', type=str, default='occluded_duke_B/base/',
                        help='path to save related informations')
    parser.add_argument('--max_save_model_num', type=int, default=1, help='0 for max num is infinit')
    parser.add_argument('--resume_train_epoch', type=int, default=-1, help='-1 for no resuming')
    parser.add_argument('--auto_resume_training_from_lastest_step', type=ast.literal_eval, default=True)
    parser.add_argument('--total_train_epoch', type=int, default=120)
    parser.add_argument('--eval_epoch', type=int, default=5)
    parser.add_argument('--resume_test_model', type=int, default=119, help='-1 for no resuming')
    parser.add_argument('--test_mode', type=str, default='inter-camera', help='inter-camera, intra-camera, all')


    config = parser.parse_args()
    seed_torch(config.seed)
    main(config)




