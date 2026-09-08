# coding=utf-8
"""Shared dataset, transform, and model definitions used by both
train.py (training) and load_model_predict.ipynb (inference)."""
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import ResNet50_Weights

# Per-dataset paths for the raw images and the pickled train/test split info.
DATASET_CONFIGS = {
    'CUB_H': {
        'root_dir': './data/CUB_200_2011/CUB_200_2011/images',
        'info_file': './data/CUB_200_2011/CUB_200_2011/CUB_200_2011_train_test_multi_level_info.pkl',
    },
    'Butterfly_H': {
        'root_dir': './data/Butterfly200/Butterfly200/images',
        'info_file': './data/Butterfly200/Butterfly200/Butterfly_train_test_multi_level_info.pkl',
    },
}


def seed_everything(seed=65):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


###########set the visible GPU cards#################
def set_gpu_envs(gpu_list):
    gpu_envs_str = ','.join(str(gpu) for gpu in gpu_list)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_envs_str
    return gpu_envs_str
###############################################


def spatial_random(imgdata, imgwith=224):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(imgwith),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                          std=(0.229, 0.224, 0.225)),
    ])
    return transform(imgdata)


def spatial_center(imgdata, imgwith=224):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(int(imgwith * 8 / 7)),  # Let smaller edge match
        torchvision.transforms.CenterCrop(imgwith),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                          std=(0.229, 0.224, 0.225)),
    ])
    return transform(imgdata)


def process_image(imgdata, resize_width, transfer_mode):
    if transfer_mode == 'random':
        imgdata = spatial_random(imgdata, resize_width)

    if transfer_mode == 'center':
        imgdata = spatial_center(imgdata, resize_width)

    return imgdata


def load_hierarchy_info(setname):
    """Load the pickled train/test file lists, hierarchical labels, and
    label-count/transition-matrix info for the given dataset."""
    with open(DATASET_CONFIGS[setname]['info_file'], 'rb') as train_test_info:
        filelist_train = pickle.load(train_test_info)
        label_train = pickle.load(train_test_info)
        filelist_test = pickle.load(train_test_info)
        label_test = pickle.load(train_test_info)
        label_len = pickle.load(train_test_info)
        _ = pickle.load(train_test_info)
        _ = pickle.load(train_test_info)
        _ = pickle.load(train_test_info)
        _ = pickle.load(train_test_info)
        _ = pickle.load(train_test_info)  # trans_0_to_1 (unused here)
        trans_1_to_2 = pickle.load(train_test_info)
        trans_2_to_3 = pickle.load(train_test_info)
        trans_3_to_4 = pickle.load(train_test_info)

    return {
        'filelist_train': filelist_train,
        'label_train': label_train,
        'filelist_test': filelist_test,
        'label_test': label_test,
        'label_len': label_len,
        'trans_1_to_2': trans_1_to_2,
        'trans_2_to_3': trans_2_to_3,
        'trans_3_to_4': trans_3_to_4,
    }


def get_fused_output(outputs_list, trans_1_to_4, trans_2_to_4, trans_3_to_4):
    """Combine the four per-granularity predictions into one fine-grained
    prediction, by mapping the coarser predictions onto the finest label
    space and summing them with the model's own finest-level prediction."""
    softmax = torch.nn.Softmax(dim=1)
    one_merged = torch.zeros_like(outputs_list[-1])
    one_merged += torch.matmul(softmax(outputs_list[0]), trans_1_to_4)
    one_merged += torch.matmul(softmax(outputs_list[1]), trans_2_to_4)
    one_merged += torch.matmul(softmax(outputs_list[2]), trans_3_to_4)
    one_merged += softmax(outputs_list[-1])
    return one_merged


class FineGrainedDataset(Dataset):
    """Hierarchical fine-grained image dataset (e.g. CUB-200-2011, Butterfly200)."""

    def __init__(self, setname, resize_width, transfer_mode, is_train=True, show_info=False):
        self.root_dir = DATASET_CONFIGS[setname]['root_dir']
        self.resize_width = resize_width
        self.is_train = is_train
        self.transfer_mode = transfer_mode

        info = load_hierarchy_info(setname)
        self.filelist_train = info['filelist_train']
        self.label_train = info['label_train']
        self.filelist_test = info['filelist_test']
        self.label_test = info['label_test']

        assert len(self.label_train) == len(self.filelist_train)
        assert len(self.label_test) == len(self.filelist_test)
        if show_info:
            print('Number of samples: {:d}, train:{}, test:{}, mode:{}'.format(
                len(self.filelist_train) + len(self.filelist_test),
                len(self.filelist_train), len(self.filelist_test), self.transfer_mode))

    def __getitem__(self, index):
        if self.is_train:
            imgfile = self.filelist_train[index]
            imglabel = self.label_train[index]
        else:
            imgfile = self.filelist_test[index]
            imglabel = self.label_test[index]
        imgdata_pil = Image.open(os.path.join(self.root_dir, imgfile)).convert('RGB')
        imgdata = process_image(imgdata_pil, self.resize_width, self.transfer_mode)
        imgdata = imgdata.float()
        imglabel = torch.tensor(imglabel).long()
        return imgdata, imglabel

    def __len__(self):
        return len(self.filelist_train) if self.is_train else len(self.filelist_test)


class ResNet(nn.Module):

    def __init__(self, features_base, feature_branch_list, fc_indim=512 * 4, grid_rate=1, num_classes=[13, 37, 122, 200]):
        super(ResNet, self).__init__()

        self.features_base = features_base
        self.features_branch_1 = feature_branch_list[0]
        self.features_branch_2 = feature_branch_list[1]
        self.features_branch_3 = feature_branch_list[2]
        self.features_branch_4 = feature_branch_list[3]

        self.fc_ori_1 = nn.Linear(fc_indim, num_classes[0])
        self.fc_ori_2 = nn.Linear(fc_indim, num_classes[1])
        self.fc_ori_3 = nn.Linear(fc_indim, num_classes[2])
        self.fc_ori_4 = nn.Linear(fc_indim, num_classes[3])

        self.averpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        b, c, w, h = x.size()
        x = self.features_base(x)

        x_out_1 = self.features_branch_1(x.detach())
        x_out_1 = self.averpool(x_out_1).view(b, -1)
        x_out_1 = self.fc_ori_1(x_out_1)

        x_out_2 = self.features_branch_2(x.detach())
        x_out_2 = self.averpool(x_out_2).view(b, -1)
        x_out_2 = self.fc_ori_2(x_out_2)

        x_out_3 = self.features_branch_3(x.detach())
        x_out_3 = self.averpool(x_out_3).view(b, -1)
        x_out_3 = self.fc_ori_3(x_out_3)

        x_out_4 = self.features_branch_4(x)
        x_out_4 = self.averpool(x_out_4).view(b, -1)
        x_out_4 = self.fc_ori_4(x_out_4)

        return [x_out_1, x_out_2, x_out_3, x_out_4]


def resnet50(pretrained=True, grid_rate=1, num_classes=[], **kwargs):
    """Constructs a hierarchical ResNet-50 model with one shared trunk (conv1-3)
    feeding four independent branches, one per label granularity.

    Args:
        pretrained (bool): If True, each branch's backbone is pre-trained on ImageNet.
    """
    features_tmp_0 = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    features_base = torch.nn.Sequential(*list(features_tmp_0.children())[:-3])  # conv1-3

    feature_branch_list = []
    for _ in range(4):
        features_tmp = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        feature_branch_list.append(torch.nn.Sequential(*list(features_tmp.children())[-3:-2]))

    model = ResNet(features_base, feature_branch_list, 512 * 4, grid_rate, num_classes, **kwargs)

    if pretrained:
        print('Load pre-trained Resnet50 success!')
    else:
        print('Load Resnet50 success!')

    return model
