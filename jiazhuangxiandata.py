from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

import random
import h5py
import torch
from scipy import ndimage
from scipy.ndimage import zoom  # from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from scipy import ndimage
from PIL import Image


# from configs.config_jiazhuangxian import setting_config # 只有debug的时候用


# ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
# class Polyp_datasets(Dataset):


class jiazhuangxian_datasets(Dataset):
    def __init__(self, path_Data, config, mode='train'):#train=True
        super(jiazhuangxian_datasets, self)
        if mode == 'train':
            images_list = sorted(os.listdir('/root/autodl-tmp/dataset/new/ddti/train/imgs/'))  # train/image/
            masks_list = sorted(os.listdir('/root/autodl-tmp/dataset/new/ddti/train/mask/'))  # path_Data+train/mask/
            self.data = []
            for i in range(len(images_list)):
                img_path = '/root/autodl-tmp/dataset/new/ddti/train/imgs/' + images_list[i]  # path_Data+'train/image'
                mask_path = '/root/autodl-tmp/dataset/new/ddti/train/mask/' + masks_list[i]  # path_Data+'train/mask'
                self.data.append([img_path, mask_path])
            self.transformer = config.train_transformer
        elif mode == 'val':
            images_list = sorted(os.listdir('/root/autodl-tmp/dataset/new/ddti/val/imgs/'))  # path_Data+'test/image/'
            masks_list = sorted(os.listdir('/root/autodl-tmp/dataset/new/ddti/val/mask/'))  # path_Data+'test/mask/'
            self.data = []
            for i in range(len(images_list)):
                img_path = '/root/autodl-tmp/dataset/new/ddti/val/imgs/' + images_list[i]  # path_Data+'test/image/'
                mask_path = '/root/autodl-tmp/dataset/new/ddti/val/mask/' + masks_list[i]  # path_Data+'test/mask/'
                self.data.append([img_path, mask_path])
            self.transformer = config.test_transformer
        elif mode == 'test':  # test 数据集需要 加 test 数据集的名称
            images_list = sorted(os.listdir('/root/autodl-tmp/dataset/new/ddti/test/imgs/'))  # path_Data+'test/image/'
            masks_list = sorted(os.listdir('/root/autodl-tmp/dataset/new/ddti/test/mask/'))  # path_Data+'test/mask/'
            self.data = []
            for i in range(len(images_list)):
                img_path = '/root/autodl-tmp/dataset/new/ddti/test/imgs/' + images_list[i]  # path_Data+'test/image/'
                mask_path = '/root/autodl-tmp/dataset/new/ddti/test/mask/' + masks_list[i]  # path_Data+'test/mask/'
                self.data.append([img_path, mask_path])
            self.transformer = config.test_transformer


# class jiazhuangxian_datasets(Dataset):
#     def __init__(self, path_Data, config, mode='train'):#train=True
#         super(jiazhuangxian_datasets, self)
#         if mode == 'train':
#             images_list = sorted(os.listdir('/root/autodl-tmp/dataset/new/tn3k/train/imgs/'))  # train/image/
#             masks_list = sorted(os.listdir('/root/autodl-tmp/dataset/new/tn3k/train/mask/'))  # path_Data+train/mask/
#             self.data = []
#             for i in range(len(images_list)):
#                 img_path = '/root/autodl-tmp/dataset/new/tn3k/train/imgs/' + images_list[i]  # path_Data+'train/image'
#                 mask_path = '/root/autodl-tmp/dataset/new/tn3k/train/mask/' + masks_list[i]  # path_Data+'train/mask'
#                 self.data.append([img_path, mask_path])
#             self.transformer = config.train_transformer
#         elif mode == 'val':
#             images_list = sorted(os.listdir('/root/autodl-tmp/dataset/new/tn3k/val/imgs/'))  # path_Data+'test/image/'
#             masks_list = sorted(os.listdir('/root/autodl-tmp/dataset/new/tn3k/val/mask/'))  # path_Data+'test/mask/'
#             self.data = []
#             for i in range(len(images_list)):
#                 img_path = '/root/autodl-tmp/dataset/new/tn3k/val/imgs/' + images_list[i]  # path_Data+'test/image/'
#                 mask_path = '/root/autodl-tmp/dataset/new/tn3k/val/mask/' + masks_list[i]  # path_Data+'test/mask/'
#                 self.data.append([img_path, mask_path])
#             self.transformer = config.test_transformer
#         elif mode == 'test':  # test 数据集需要 加 test 数据集的名称
#             images_list = sorted(os.listdir('/root/autodl-tmp/dataset/new/tn3k/test/imgs/'))  # path_Data+'test/image/'
#             masks_list = sorted(os.listdir('/root/autodl-tmp/dataset/new/tn3k/test/mask/'))  # path_Data+'test/mask/'
#             self.data = []
#             for i in range(len(images_list)):
#                 img_path = '/root/autodl-tmp/dataset/new/tn3k/test/imgs/' + images_list[i]  # path_Data+'test/image/'
#                 mask_path = '/root/autodl-tmp/dataset/new/tn3k/test/mask/' + masks_list[i]  # path_Data+'test/mask/'
#                 self.data.append([img_path, mask_path])
#             self.transformer = config.test_transformer
    ##如果需要增加val在这增加一个更test类似结构的就行


    def __getitem__(self, index):
        img_path, msk_path = self.data[index]
        img = np.array(Image.open(img_path).convert('RGB'))
        # isic 数据集未做二值化处理
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):

        return len(self.data)


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample
