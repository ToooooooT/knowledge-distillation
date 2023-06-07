from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
from mmseg.datasets import ADE20KDataset
from mmengine import Config

default_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

class ade20k_dataset(Dataset):
    def __init__(self, args, mode='train', transform=default_transform):
        cfg = Config.fromfile(args.config)
        self.root = './data/ADEChallengeData2016'
        self.mode = mode
        if mode == 'train':
            self.data = ADE20KDataset(data_root=self.root,
                                  data_prefix=dict(img_path='images/training', seg_map_path='annotations/training'),
                                  img_suffix='.jpg',
                                  seg_map_suffix='.png', 
                                  pipeline=cfg.get('train_pipeline'))
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
        transform = transforms.Compose([
            transforms.Resize(640, 640),
        ])
        return transform(img)
    
    def get_label(self, index):
        # TODO: fix this function to load annotation image
        # label = self.data[index]['gt_semantic_seg']
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
