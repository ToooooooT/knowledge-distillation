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

from mmengine import Config
from mmengine.runner import Runner
from mmseg.apis import inference_model
from mmseg.models import build_segmentor

from decode_heads import atm_head
from dataset.ade20k_dataset import ade20k_dataset

from losses.atm_loss import ATMLoss

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
    parser.add_argument('--img_size', default=640, type=int, help='the image size')
    parser.add_argument('--encoder_in_channels', default=3, type=int, help='the input channels of encoder')
    parser.add_argument('--encoder_embed_dims', default=1024, type=int, help='the embedding dimension of encoder')
    parser.add_argument('--encoder_num_layers', default=6, type=int, help='number of layers of encoder')
    parser.add_argument('--encoder_num_heads', default=8, type=int, help='number of head of encoder')
    # out_indices (list | int) Output from which stages.
    parser.add_argument('--drop_path_rate', default=0., type=float, help='stochastic depth rate')
    # ---------------------- decoder ----------------------
    parser.add_argument('--single', default=False, action='store_true')
    parser.add_argument('--decoder_in_channels', default=1024, type=int, help='the input channels of decoder')
    parser.add_argument('--decoder_embed_dims', default=512, type=int, help='the embedding dimensions of decoder')
    parser.add_argument('--decoder_num_heads', default=8, type=int, help='number of head of decoder')
    # use_stages = len(out_indices).

    args = parser.parse_args()
    return args

def train(teacher, student, img, label, device):
    img = img.to(device)
    label = label.to(device)
    with torch.no_grad():
        teacher_pred = teacher(img)
        student_pred = student(img)
    loss_module = ATMLoss(num_classes=150, dec_layers=len(student.encoder.out_indices), loss_weight=1.0)
    loss_module(teacher_pred, label, 0)

def main():
    args = parse_args()

    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
    else:
        assert False, 'Using CPU.'
    
    out_indices = [2, 5]
    use_stages = 2

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
                                     use_stages=use_stages,
                                     single=args.single
                                     )
    teacher = build_segmentor(cfg=cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = torch.load(args.checkpoint)
    del checkpoint['state_dict']['decode_head.loss_decode.criterion.empty_weight']
    teacher.load_state_dict(checkpoint['state_dict'])

    # --------- transfer to device ------------------------------------
    simple_segvit = simple_segvit.to(device)
    teacher = teacher.to(device)

    # --------- load a dataset ------------------------------------
    # TODO: build dataset
    train_data = ade20k_dataset(args.config, mode='train')
    train_loader = DataLoader(train_data,
                            num_workers=1,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)
    valid_data = ade20k_dataset(args.config, mode='valid')
    valid_loader = DataLoader(valid_data,
                            num_workers=1,
                            batch_size=1,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    # for img, label in train_loader:
    #     break

    for img, label in valid_loader:
        train(teacher, simple_segvit, img, label, device)
        break
    

    # --------- optimizers ------------------------------------
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam
    elif args.optimizer == 'rmsprop':
        args.optimizer = optim.RMSprop
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD
    else:
        raise ValueError(f'Unknown optimizer: {args.optimizer}')

    # optimizer = args.optimizer(simple_segvit.parameters(), lr=args.lr)

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