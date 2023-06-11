from torch.utils.data import Dataset
import torch
from torchvision import transforms
import torchvision
from mmseg.datasets import ADE20KDataset
from mmengine import Config

default_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

class ade20k_dataset(Dataset):
    def __init__(self, config_path, mode='train', root='./data/ADEChallengeData2016', crop_size=(640, 640), isTeach=True):
        cfg = Config.fromfile(config_path)
        self.root = root
        self.mode = mode
        self.crop_size = crop_size
        self.isTeach = isTeach
        if mode == 'train':
            self.data = ADE20KDataset(data_root=self.root,
                                  data_prefix=dict(img_path='images/training', seg_map_path='annotations/training'),
                                  img_suffix='.jpg',
                                  seg_map_suffix='.png', 
                                  pipeline=cfg.get('test_pipeline'))
        elif mode == 'valid':
            self.data = ADE20KDataset(data_root=self.root,
                                  data_prefix=dict(img_path='images/validation', seg_map_path='annotations/validation'),
                                  img_suffix='.jpg',
                                  seg_map_suffix='.png', 
                                  pipeline=cfg.get('test_pipeline'))
            
    def __len__(self):
        return len(self.data)
        
    def get_img(self, index):
        img = self.data[index]['img']
        if self.mode == 'train':
            if img.shape[1] < self.crop_size[0] and img.shape[2] < self.crop_size[1]:
                transform = transforms.Compose([
                    transforms.Resize(self.crop_size),
                    transforms.RandomCrop(self.crop_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                ])
            elif img.shape[1] < self.crop_size[0]:
                transform = transforms.Compose([
                    transforms.Resize((self.crop_size[0], img.shape[2])),
                    transforms.RandomCrop(self.crop_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                ])
            elif img.shape[2] < self.crop_size[1]:
                transform = transforms.Compose([
                    transforms.Resize((img.shape[1], self.crop_size[1])),
                    transforms.RandomCrop(self.crop_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                ])
            else:
                transform = transforms.Compose([
                    transforms.RandomCrop(self.crop_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                ])
        else:
            transform = transforms.Compose([
                transforms.Resize(self.crop_size, antialias=True)
            ])
        return transform(img)
    
    def get_label(self, index):
        label_path = self.data[index]['seg_map_path']
        label = torchvision.io.read_image(label_path)
        transform = transforms.Compose([
            transforms.Resize(self.crop_size, antialias=True)
        ])
        return transform(label)

    def get_teach_pred(self, index):
        path = self.data[index]['img_path']
        # dest_path = path.split('/')
        # dest_path[3] = 'annotations'
        # dest_path[4] = 'teacher_label'
        # dest_path[5] = dest_path[5][:-3] + 'pth'
        # dest_path = '/'.join(dest_path)
        # teach_pred = torch.load(dest_path)
        return path

    def __getitem__(self, index):
        if self.mode == 'train' or self.mode == 'valid':
            img = self.get_img(index)
            label =  self.get_label(index)
            if self.isTeach:
                teach_pred = self.get_teach_pred(index)
                return img, label, teach_pred
            return img, label
        else:
            img =  self.get_img(index)
            return img
