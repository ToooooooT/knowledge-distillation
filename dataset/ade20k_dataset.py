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
    def __init__(self, config_path, mode='train', root='./data/ADEChallengeData2016', crop_size=(640, 640)):
        cfg = Config.fromfile(config_path)
        self.root = root
        self.mode = mode
        self.crop_size = crop_size
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
            transform = transforms.Compose([
                transforms.RandomCrop(self.crop_size),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((640, 640), antialias=True)
            ])
        img = transform(img)
        print(img.shape)
        print(type(img))
        return img
    
    def get_label(self, index):
        label_path = self.data[index]['seg_map_path']
        label = torchvision.io.read_image(label_path)
        return label

    def __getitem__(self, index):
        if self.mode == 'train' or self.mode == 'valid':
            img = self.get_img(index)
            label =  self.get_label(index)
            return img, label
        else:
            img =  self.get_img(index)
            return img
