import argparse
import os
import random
from tqdm import tqdm

import numpy as np

from models.SimpleSegViT.SimpleSegViT import SimpleSegViT
from models.UNet.AttUNet import AttUNet
from models.UNet.UNet import UNet
from torchsummary import summary

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms

from mmengine import Config
from mmengine.runner import Runner
from mmseg.apis import inference_model
from mmseg.models import build_segmentor

from decode_heads import atm_head
from dataset.ade20k_dataset import ade20k_dataset

from losses.atm_loss import ATMLoss
from utils.util import intersectionAndUnionGPU 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/segvit/segvit_vit-l_jax_640x640_160k_ade20k.py')
    parser.add_argument('--checkpoint', default='./pretrained/vit_large_ade.pth')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--cuda', default=False, action='store_true')  
    parser.add_argument('--epochs', default=50, type=int, help='number of epochs to trainkj')  
    parser.add_argument('--eval_freq', default=1, type=int, help='test frequency')  
    parser.add_argument('--model_dir', default='', type=str, help='model directory')  
    parser.add_argument('--model_type', default='AttUnet', type=str, help='student model type')  
    parser.add_argument('--log_dir', default='./logs/', type=str, help='base directory to save logs')  
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
    parser.add_argument('--teach_w', default=0.5, type=float, help='the loss weight of teacher label')
    parser.add_argument('--ground_w', default=0.5, type=float, help='the loss weight of ground truth')
    parser.add_argument('--num_class', default=151, type=int, help='number of classes')

    args = parser.parse_args()
    return args


def train(teacher: nn.Module, student: nn.Module, img, label, optimizer, use_stages, args):
    # inference teacher prediction
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), antialias=True)
    ])

    with torch.no_grad():
        teacher_pred = teacher(img)
        teacher_pred = transform(teacher_pred)
        teacher_pred = F.softmax(teacher_pred, dim=1)

        B, C, H, W = img.shape
        teacher_pred = torch.cat([torch.zeros((B, 1, H, W)).to('cuda').to(torch.float), teacher_pred], dim=1)
        # Create a mask where the label is 0, shape will be [1, H, W]
        mask = (label == 0)
        teacher_pred[:, 0, :, :] = (teacher_pred[:, 0, :, :].to(torch.int) | mask.squeeze().to(torch.int)).to(torch.float)
        teacher_pred[:, 1:, :, :] = (~mask.squeeze()).to(torch.int) * teacher_pred[:, 1:, :, :]

    # TODO: segvit loss function from paper
    if isinstance(student, tuple):
        student1, student2 = student
        student1_pred = student1(img)
        student2_pred = student2(img)
        loss1 = nn.CrossEntropyLoss(student1_pred.view(-1, 150), label.view(-1)) * args.ground_w + \
                nn.CrossEntropyLoss(student1_pred.view(-1, 150), teacher_pred.view(-1)) * args.teach_w
        loss2 = nn.CrossEntropyLoss(student2_pred['pred'].view(-1, 150), label.view(-1)) * args.ground_w + \
                nn.CrossEntropyLoss(student2_pred['pred'].view(-1, 150), teacher_pred.view(-1)) * args.teach_w
    else:
        if args.model_type != 'simple_segvit':
            student_pred = student(img)
            student_pred = student_pred.permute(0, 2, 3, 1).contiguous().view(-1, 151)
            label = label.permute(0, 2, 3, 1).contiguous().view(-1)
            teacher_pred = teacher_pred.permute(0, 2, 3, 1).contiguous().view(-1, 151)
            loss = F.cross_entropy(student_pred, label) * args.ground_w + \
                    F.cross_entropy(student_pred, teacher_pred) * args.teach_w
        else:
            # get the output of simple_segvit, which will output a dict of tensors,
            # and bulid ATM module to obtain losses
            student_pred = student(img)
            # TODO: setting of ATMLoss
            atm_loss = ATMLoss(num_classes=151,
                                    dec_layers=use_stages,
                                    mask_weight=20.0,
                                    dice_weight=1.0,
                                    cls_weight=1.0,
                                    loss_weight=1.0)
            ground_losses = atm_loss.forward(student_pred, label, ignore_index=0)
            ground_losses = ground_losses['loss_ce'] + ground_losses['loss_mask'] + ground_losses['loss_dice']
            teacher_losses = F.cross_entropy(student_pred['pred_logits'].permute(0, 2, 3, 1).contiguous().view(-1, 151), teacher_pred['pred_logits'].permute(0, 2, 3, 1).contiguous().view(-1, 151)) + \
                                F.mse_loss(student_pred['pred_masks'], teacher_pred['pred_masks'])
            # combine two kinds of losses; dtpye=torch.tensor
            loss = ground_losses * args.ground_w + teacher_losses * args.teacher_w 

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach().cpu().item()


def pred(student, img, label, args):
    with torch.no_grad():
        if isinstance(student, tuple):
            student1, student2 = student
            student1_pred = student1(img)
            student2_pred = student2(img)
        else:
            # get the output of simple_segvit, which will output a dict of tensors
            student_pred = student(img)
    
    avg_mIOU = 0
    for i in range(label.shape[0]):
        i, u, _ = intersectionAndUnionGPU(student_pred[i].argmax(dim=0, keepdim=True), label[i], 151, ignore_index=0)
        mIOU = i.sum() / u.sum() # mean IOU, taking (i/u).mean() is wrong
        avg_mIOU += mIOU
    return avg_mIOU / label.shape[0]


def main():
    args = parse_args()

    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
        enable_amp = True
    else:
        enable_amp = False
        assert False, 'Using CPU.'

    assert args.teach_w + args.ground_w == 1.
    
    out_indices = [2, 5]
    use_stages = 2

    assert use_stages == len(out_indices)

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
        args.log_dir = f'{args.log_dir}/{args.model_type}/{name}'
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
    if args.model_dir != '':
        simple_segvit = saved_model['simple_segvit']
    elif args.model_type == 'AttUnet':
        student = AttUNet().to(device)
    elif args.model_type == 'Unet':
        student = UNet().to(device)
    elif args.model_type == 'simple_segvit':
        student = SimpleSegViT(shrink_idx=args.shrink_idx,
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
                                ).to(device)
    elif args.model_type == 'fuse':
        student1 = AttUNet().to(device)
        student2 = SimpleSegViT(shrink_idx=args.shrink_idx,
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
                                ).to(device)
    else:
        raise ValueError
    teacher = build_segmentor(cfg=cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = torch.load(args.checkpoint)
    del checkpoint['state_dict']['decode_head.loss_decode.criterion.empty_weight']
    teacher.load_state_dict(checkpoint['state_dict'])

    # --------- transfer to device ------------------------------------
    teacher = teacher.to(device)

    # --------- load a dataset ------------------------------------
    train_data = ade20k_dataset(args.config, mode='train', crop_size=(args.img_size, args.img_size), isTeach=False)
    train_loader = DataLoader(train_data,
                            num_workers=32,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=False)
    valid_data = ade20k_dataset(args.config, mode='valid', crop_size=(args.img_size, args.img_size), isTeach=False)
    valid_loader = DataLoader(valid_data,
                            num_workers=32,
                            batch_size=1,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=False)

    # --------- optimizers ------------------------------------
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam
    elif args.optimizer == 'rmsprop':
        args.optimizer = optim.RMSprop
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD
    else:
        raise ValueError(f'Unknown optimizer: {args.optimizer}')

    if args.model_type == 'fuse':
        optimizer = args.optimizer([student1.parameters()] + [student2.parameters()], lr=args.lr)
    else:
        optimizer = args.optimizer(student.parameters(), lr=args.lr)

    # --------- training loop ------------------------------------
    # teacher = teacher.eval()
    if args.model_type == 'fuse':
        optimizer = args.optimizer([student1.parameters()] + [student2.parameters()], lr=args.lr)
        best_test_mIoU1 = best_test_mIoU2 = 0
    else:
        optimizer = args.optimizer(student.parameters(), lr=args.lr)
        best_test_mIoU = 0

    for epoch in tqdm(range(start_epoch, epochs + 1)):
        if args.model_type == 'fuse':
            student1.train()
            student2.train()
            epoch_loss1 = epoch_loss2 = 0
        else:
            student.train()
            epoch_loss = 0

        for img, label in tqdm(train_loader):
            img = img.to(device)
            label = label.to(device)
            if args.model_type == 'fuse':
                loss1, loss2 = train(teacher, (student1, student2), img, label, optimizer, use_stages, args)
                epoch_loss1 += loss1
                epoch_loss2 += loss2
            else:
                loss = train(teacher, student, img, label, optimizer, use_stages, args)
                epoch_loss += loss

        with open(f'./{args.log_dir}/train_record.txt', 'a') as f:
            if args.model_type == 'fuse':
                f.write(f'[epoch: {epoch:02d}] loss1: {epoch_loss1:.5f} | loss2: {epoch_loss2:.5f}]\n')
            else:
                f.write(f'[epoch: {epoch:02d}] loss: {epoch_loss:.5f}]\n')

        if epoch % args.eval_freq == 0:
            # --------- testing loop ------------------------------------
            if args.model_type == 'fuse':
                student1.eval()
                student2.eval()
                test_mIoU1 = test_mIoU2 = 0
            else:
                student.eval()
                test_mIoU = 0
            for img, label in valid_loader:
                img = img.to(device)
                label = label.to(device)
                if args.model_type == 'fuse':
                    mIoU1, mIoU2 = pred((student1, student2), img, label, args)
                    test_mIoU1 += mIoU1
                    test_mIoU2 += mIoU2
                else:
                    mIoU = pred(student, img, label, args)
                    test_mIoU += mIoU

            with open(f'./{args.log_dir}/train_record.txt', 'a') as f:
                if args.model_type == 'fuse':
                    avg_test_mIoU1 = test_mIoU1 / len(valid_loader)
                    avg_test_mIoU2 = test_mIoU2 / len(valid_loader)
                    f.write(f'==================== valid mIoU1 = {avg_test_mIoU1:.5f}, mIoU2 = {avg_test_mIoU2:.5f} ====================\n')
                else:
                    avg_test_mIoU = test_mIoU / len(valid_loader)
                    f.write(f'==================== valid mIoU = {avg_test_mIoU:.5f} ====================\n')

            if args.model_type == 'fuse':
                if avg_test_mIoU1 > best_test_mIoU1:
                    best_test_mIoU1 = avg_test_mIoU1
                    torch.save({
                        'AttUnet' : student1,
                        'args': args,
                        'last_epoch': epoch},
                        f'{args.log_dir}/test_model1.pth'
                    )
                if avg_test_mIoU2 > best_test_mIoU2:
                    best_test_mIoU2 = avg_test_mIoU2
                    torch.save({
                        'simple_segvit' : student2,
                        'args': args,
                        'last_epoch': epoch},
                        f'{args.log_dir}/test_model2.pth'
                    )
            else:
                if avg_test_mIoU > best_test_mIoU:
                    best_test_mIoU = avg_test_mIoU
                    torch.save({
                        f'{args.model_type}' : student,
                        'args': args,
                        'last_epoch': epoch},
                        f'{args.log_dir}/test_model.pth'
                    )


if __name__ == '__main__':
    main()