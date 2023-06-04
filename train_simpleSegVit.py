import argparse
import os
import random
import tqdm

from models.SimpleSegViT.SimpleSegViT import SimpleSegViT
from torchsummary import summary

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import mmcv
from mmengine import Config
from mmengine.runner import Runner
from mmseg.apis import inference_model
from mmseg.models import build_segmentor
from mmseg.datasets import ADE20KDataset

from decode_heads import atm_head

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/segvit/segvit_vit-l_jax_640x640_160k_ade20k.py')
    parser.add_argument('--checkpoint', default='./pretrained/vit_large_ade.pth')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--cuda', default=False, action='store_true')  
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs to trainkj')  
    parser.add_argument('--eval_freq', default=4, type=int, help='test frequency')  
    parser.add_argument('--model_dir', default='', type=str, help='model directory')  
    parser.add_argument('--log_dir', default='./logs/SimpleSegVit', type=str, help='base directory to save logs')  
    parser.add_argument('--optimizer', default='adam', type=str, help='optimizer to train with')  
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    # ---------------------- encoder ----------------------
    parser.add_argument('--shrink_idx', default=-1, type=int, help='the layer to shrink')
    parser.add_argument('--img_size', default=512, type=int, help='the image size')
    parser.add_argument('--encoder_in_channels', default=3, type=int, help='the input channels of encoder')
    parser.add_argument('--encoder_embed_dims', default=1024, type=int, help='the embedding dimension of encoder')
    parser.add_argument('--encoder_num_layers', default=12, type=int, help='number of layers of encoder')
    parser.add_argument('--encoder_num_heads', default=8, type=int, help='number of head of encoder')
    # out_indices (list | int) Output from which stages.
    parser.add_argument('--drop_path_rate', default=0., type=float, help='stochastic depth rate')
    # ---------------------- decoder ----------------------
    parser.add_argument('--decoder_in_channels', default=1024, type=int, help='the input channels of decoder')
    parser.add_argument('--decoder_embed_dims', default=512, type=int, help='the embedding dimensions of decoder')
    parser.add_argument('--decoder_num_heads', default=8, type=int, help='number of head of decoder')
    # use_stages = len(out_indices).

    args = parser.parse_args()
    return args


def train():
    pass

def main():
    args = parse_args()

    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
    else:
        assert False, 'Using CPU.'
    
    out_indices = -1
    use_stages = 1

    if args.model_dir != '':
        # load model and continue training from checkpoint
        saved_model = torch.load('%s/model.pth' % args.model_dir)
        optimizer = args.optimizer
        model_dir = args.model_dir
        epochs = args.epochs
        args = saved_model['args']
        args.optimizer = optimizer
        args.model_dir = model_dir
        args.log_dir = f'{args.log_dir}/continued'
        start_epoch = saved_model['last_epoch']
    else:
        name = f'lr={args.lr}-batch_size={args.batch_size}'
        args.log_dir = f'{args.log_dir}/{name}'
        epochs = args.epochs
        start_epoch = 1

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs('%s/gen/' % args.log_dir, exist_ok=True)

    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    if os.path.exists(f'./{args.log_dir}/train_record.txt'):
        os.remove(f'./{args.log_dir}/train_record.txt')
    
    print(args)

    with open(f'./{args.log_dir}/train_record.txt', 'a') as train_record:
        train_record.write('args: {}\n'.format(args))

    cfg = Config.fromfile(args.config)

    # ------------ build the models  --------------
    # TODO: build teacher model
    if args.model_dir != '':
        simple_segvit = saved_model['simple_segvit']
    else:
        simple_segvit = SimpleSegViT(shrink_idx=args.shrink_idx,
                                     img_size=args.img_size,
                                     encoder_in_channels=args.encoder_in_channels,
                                     encoder_embed_dims=args.encoder_embed_dims,
                                     encoder_num_layers=args.encoder_num_layers,
                                     encoder_num_heads=args.encoder_num_heads,
                                     drop_path_rate=args.drop_path_rate,
                                     decoder_in_channels=args.decoder_in_channels,
                                     decoder_embed_dims=args.decoder_embed_dims,
                                     decoder_num_heads=args.decoder_num_heads,
                                     out_indices=out_indices,
                                     use_stages=use_stages
                                     )
    teacher = build_segmentor(cfg=cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = torch.load(args.checkpoint)
    del checkpoint['state_dict']['decode_head.loss_decode.criterion.empty_weight']
    teacher.load_state_dict(checkpoint['state_dict'])

    # --------- transfer to device ------------------------------------
    simple_segvit = simple_segvit.to(device)
    teacher = teacher.to(device)
    # teacher.to(device)

    # --------- load a dataset ------------------------------------
    # TODO: build dataset
    # train_dataset = ADE20KDataset(data_root='./data/ADEChallengeData2016',
    #                               data_prefix=dict(img_path='images/training', seg_map_path='annotations/training'),
    #                               img_suffix='.jpg',
    #                               seg_map_suffix='.png', pipeline=cfg.get('test_pipeline'))
    valid_dataset = ADE20KDataset(data_root='./data/ADEChallengeData2016',
                                  data_prefix=dict(img_path='images/validation', seg_map_path='annotations/validation'),
                                  img_suffix='.jpg',
                                  seg_map_suffix='.png', 
                                  pipeline=cfg.get('test_pipeline'))
    
    
    # valid_loader = DataLoader(dataset=valid_dataset, 
    #                           batch_size=args.batch_size, 
    #                           shuffle=False, 
    #                           num_workers=32, 
    #                           pin_memory=True,
    #                           drop_last=False)


    # --------- optimizers ------------------------------------
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam
    elif args.optimizer == 'rmsprop':
        args.optimizer = optim.RMSprop
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD
    else:
        raise ValueError(f'Unknown optimizer: {args.optimizer}')

    optimizer = args.optimizer(simple_segvit.parameters(), lr=args.lr)

    # --------- training loop ------------------------------------
    # TODO
    # best_test_acc = best_new_test_acc = 0
    # for epoch in tqdm(range(start_epoch, epochs + 1)):
    #     simple_segvit.train()

    #     epoch_loss = 0

    #     for img in train_loader:
    #         loss = train()

    #     with open(f'./{args.log_dir}/train_record.txt', 'a') as f:
    #         f.write(f'[epoch: {epoch:02d}] loss: {epoch_loss:.5f}\n')

    #     if epoch % args.eval_freq == 0:
    #         # --------- testing loop ------------------------------------
    #         simple_segvit.eval()
    #         test_acc = 0
    #         for img in valid_loader:
    #             pass

    #         avg_test_acc = test_acc / len(test_loader)

    #         with open(f'./{args.log_dir}/train_record.txt', 'a') as f:
    #             f.write(f'==================== valid mIoU = {avg_test_acc:.5f} ====================\n')

    #         if avg_test_acc > best_test_acc:
    #             best_test_acc = avg_test_acc
    #             torch.save({
    #                 'simple_segvit' : simple_segvit,
    #                 'args': args,
    #                 'las_epoch': epoch},
    #                 f'{args.log_dir}/test_model.pth'
    #             )


if __name__ == '__main__':
    main()