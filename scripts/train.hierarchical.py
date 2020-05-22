#!/usr/bin/env python

import os
gid = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(gid)

import time
import pickle
import random
from collections import OrderedDict

import sys
sys.path.append('..')
from src.training_manager import TrainingManager, get_current_time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils import spectral_norm

torch.multiprocessing.set_sharing_strategy('file_system')


class VocDataset(Dataset):
    def __init__(self, ids, path):
        self.metadata = ids
        self.path = path

    def __getitem__(self, index):
        id = self.metadata[index]
        voc_fp = os.path.join(self.path, id, 'vocals.npy')

        voc = np.load(voc_fp)

        return voc

    def __len__(self):
        return len(self.metadata)


def get_voc_datasets(path, feat_type, batch_size, va_samples):

    dataset_fp = os.path.join(path, f'dataset.pkl')
    in_dir = os.path.join(path, feat_type)
    with open(dataset_fp, 'rb') as f:
        dataset = pickle.load(f)

    dataset_ids = [x[0] for x in dataset]

    random.seed(1234)
    random.shuffle(dataset_ids)

    va_ids = dataset_ids[-va_samples:]
    tr_ids = dataset_ids[:-va_samples]

    tr_dataset = VocDataset(tr_ids, in_dir)
    va_dataset = VocDataset(va_ids, in_dir)
    num_tr = len(tr_dataset)
    num_va = len(va_dataset)

    iterator_tr = DataLoader(
        tr_dataset,
        batch_size=batch_size,
        num_workers=5,
        shuffle=True,
        drop_last=True,
        pin_memory=True)

    iterator_va = DataLoader(
        va_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        drop_last=True,
        pin_memory=True)

    return iterator_tr, num_tr, iterator_va, num_va


def validate():
    # Store random state
    cpu_rng_state_tr = torch.get_rng_state()
    gpu_rng_state_tr = torch.cuda.get_rng_state()

    # Set random stae
    torch.manual_seed(123)

    # ###
    sum_losses_va = OrderedDict([(loss_name, 0) for loss_name in loss_funcs])

    count_all_va = 0

    # In validation, set netG.eval()
    netG.eval()
    netD.eval()
    num_batches_va = len(iterator_va)
    with torch.set_grad_enabled(False):
        for i_batch, batch in enumerate(iterator_va):

            # voc.shape=(bs, feat_dim, num_frames)
            voc = batch
            voc = voc.cuda()
            voc = (voc - mean) / std

            bs, _, nf = voc.size()

            # ### Train generator ###
            z = torch.zeros((bs, z_dim, int(np.ceil(nf/z_total_scale_factor)))).normal_(0, 1).float().cuda()

            fake_voc = netG(z)

            gloss = torch.mean(torch.abs(netD(fake_voc)-fake_voc))

            # ### Train discriminator ###
            real_dloss = torch.mean(torch.abs(netD(voc)-voc))
            fake_dloss = torch.mean(torch.abs(netD(fake_voc.detach())-fake_voc.detach()))

            dloss = real_dloss - k * fake_dloss

            # ### Convergence ###
            _, convergence = recorder(real_dloss, fake_dloss, update_k=False)

            # ### Losses ###
            losses = OrderedDict([
                ('G', gloss),
                ('D', dloss),
                ('RealD', real_dloss),
                ('FakeD', fake_dloss),
                ('Convergence', convergence),
            ])

            # ### Misc ###
            count_all_va += bs

            # Accumulate losses
            losses_va = OrderedDict([(loss_name, lo.item()) for loss_name, lo in losses.items()])

            for loss_name, lo in losses_va.items():
                sum_losses_va[loss_name] += lo*bs

            if i_batch % 10 == 0:
                print('{}/{}'.format(i_batch, num_batches_va))
    mean_losses_va = OrderedDict([(loss_name, slo / count_all_va) for loss_name, slo in sum_losses_va.items()])

    # Restore rng state
    torch.set_rng_state(cpu_rng_state_tr)
    torch.cuda.set_rng_state(gpu_rng_state_tr)

    return mean_losses_va


def make_inf_iterator(data_iterator):
    while True:
        for data in data_iterator:
            yield data


class BEGANRecorder(nn.Module):
    def __init__(self, lambda_k, init_k, gamma):
        super().__init__()
        self.lambda_k = lambda_k
        self.init_k = init_k
        self.gamma = gamma

        self.k = nn.Parameter(torch.tensor(init_k))

    def forward(self, real_dloss, fake_dloss, update_k=False):
        # convergence
        diff = self.gamma * real_dloss - fake_dloss
        convergence = (real_dloss + torch.abs(diff))

        # Update k
        if update_k:
            self.k.data = torch.clamp(self.k + self.lambda_k * diff, 0, 1).data

        return self.k.item(), convergence


class RCBlock(nn.Module):
    def __init__(self, feat_dim, ks, dilation, num_groups):
        super().__init__()
        ksm1 = ks-1
        mfd = feat_dim
        di = dilation
        self.num_groups = num_groups

        self.relu = nn.LeakyReLU()

        self.rec = nn.GRU(mfd, mfd, num_layers=1, batch_first=True, bidirectional=True)
        self.conv = nn.Conv1d(mfd, mfd, ks, 1, ksm1*di//2, dilation=di, groups=num_groups)
        self.gn = nn.GroupNorm(num_groups, mfd)

    def init_hidden(self, batch_size, hidden_size):
        num_layers = 1
        num_directions = 2
        hidden = torch.zeros(num_layers*num_directions, batch_size, hidden_size)
        hidden.normal_(0, 1)
        return hidden

    def forward(self, x):
        bs, mfd, nf = x.size()

        hidden = self.init_hidden(bs, mfd).to(x.device)

        r = x.transpose(1, 2)
        r, _ = self.rec(r, hidden)
        r = r.transpose(1, 2).view(bs, 2, mfd, nf).sum(1)
        c = self.relu(self.gn(self.conv(r)))
        x = x+r+c

        return x


class BodyGBlock(nn.Module):
    def __init__(self, input_dim, output_dim, middle_dim, num_groups):
        super().__init__()

        ks = 3  # kernel size
        mfd = middle_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mfd = mfd
        self.num_groups = num_groups

        # ### Main body ###
        block = [
            nn.Conv1d(input_dim, mfd, 3, 1, 1),
            nn.GroupNorm(num_groups, mfd),
            nn.LeakyReLU(),
            RCBlock(mfd, ks, dilation=1, num_groups=num_groups),
            nn.Conv1d(mfd, output_dim, 3, 1, 1),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):

        # ### Main ###
        x = self.block(x)

        return x


class NetG(nn.Module):
    def __init__(self, feat_dim, z_dim, z_scale_factors):
        super().__init__()

        mfd = 512
        num_groups = 4
        self.num_groups = num_groups
        self.mfd = mfd

        self.feat_dim = feat_dim
        self.z_dim = z_dim
        self.z_scale_factors = z_scale_factors

        # ### Main body ###
        self.block0 = BodyGBlock(z_dim, mfd, mfd, num_groups)
        self.head0 = nn.Conv1d(mfd, feat_dim, 3, 1, 1)

        blocks = []
        heads = []
        for scale_factor in z_scale_factors:
            block = BodyGBlock(mfd, mfd, mfd, num_groups)
            blocks.append(block)

            head = nn.Conv1d(mfd, feat_dim, 3, 1, 1)
            heads.append(head)

        self.blocks = nn.ModuleList(blocks)
        self.heads = nn.ModuleList(heads)

    def forward(self, z):

        # SBlock0
        z_scale_factors = self.z_scale_factors
        x_body = self.block0(z)
        x_head = self.head0(x_body)

        for ii, (block, head, scale_factor) in enumerate(zip(self.blocks, self.heads, z_scale_factors)):
            x_body = F.interpolate(x_body, scale_factor=scale_factor, mode='nearest')
            x_head = F.interpolate(x_head, scale_factor=scale_factor, mode='nearest')

            x_body = x_body + block(x_body)

            x_head = x_head + head(x_body)

        return x_head


class BNSNConv2dDBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, frequency_stride, time_dilation):
        super().__init__()
        ks = kernel_size
        ksm1d2 = (ks - 1) // 2

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.time_dilation = time_dilation
        self.frequency_stride = frequency_stride

        block = [
            spectral_norm(nn.Conv2d(
                input_dim, output_dim, ks,
                (frequency_stride, 1),
                (1, time_dilation*ksm1d2),
                dilation=(1, time_dilation))),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):

        x = self.block(x)

        return x


class BNSNConv1dDBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, dilation):
        super().__init__()
        ks = kernel_size
        ksm1d2 = (ks - 1) // 2

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.dilation = dilation

        block = [
            spectral_norm(nn.Conv1d(input_dim, output_dim, ks,
                                    1, dilation*ksm1d2, dilation=dilation)),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):

        x = self.block(x)

        return x


class StridedBNSNConv1dDBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride):
        super().__init__()
        ks = kernel_size
        ksm1d2 = (ks - 1) // 2

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.stride = stride

        block = [
            spectral_norm(nn.Conv1d(input_dim, output_dim, ks, stride, ksm1d2)),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):

        x = self.block(x)

        return x


class NetD(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        ks = 3  # kernel size
        mfd = 512

        self.mfd = mfd

        self.input_size = input_size

        # ### Main body ###
        blocks2d = [
            BNSNConv2dDBlock(1, 4, ks, 2, 2),
            BNSNConv2dDBlock(4, 16, ks, 2, 4),
            BNSNConv2dDBlock(16, 64, ks, 2, 8),
        ]

        blocks1d = [
            BNSNConv1dDBlock(64*10, mfd, 3, 1),
            BNSNConv1dDBlock(mfd, mfd, ks, 16),
            BNSNConv1dDBlock(mfd, mfd, ks, 32),
            BNSNConv1dDBlock(mfd, mfd, ks, 64),
            BNSNConv1dDBlock(mfd, mfd, ks, 128),
        ]

        self.body2d = nn.Sequential(*blocks2d)
        self.body1d = nn.Sequential(*blocks1d)

        self.head = spectral_norm(nn.Conv1d(mfd, input_size, 3, 1, 1))

    def forward(self, x):
        '''
        x.shape=(batch_size, feat_dim, num_frames)
        cond.shape=(batch_size, cond_dim, num_frames)
        '''
        bs, fd, nf = x.size()

        # ### Process generated ###
        # shape=(bs, 1, fd, nf)
        x = x.unsqueeze(1)

        # shape=(bs, 64, 10, nf_)
        x = self.body2d(x)
        # shape=(bs, 64*10, nf_)
        x = x.view(bs, -1, x.size(3))

        # ### Merging ###
        x = self.body1d(x)

        # ### Head ###
        # shape=(bs, input_size, nf)
        # out = torch.sigmoid(self.head(x))
        out = self.head(x)

        # Pad
        # out = F.pad(out, pad=(0, nf-out.size(2)))

        return out


class Encoder(nn.Module):
    def __init__(self, input_size, z_dim, z_scale_factors):
        super().__init__()

        ks = 3  # kernel size
        mfd = 512

        self.mfd = mfd

        self.input_size = input_size
        self.z_dim = z_dim
        self.z_scale_factors = z_scale_factors

        # ### Main body ###
        blocks2d = [
            BNSNConv2dDBlock(1, 4, ks, 2, 2),
            BNSNConv2dDBlock(4, 16, ks, 2, 4),
            BNSNConv2dDBlock(16, 64, ks, 2, 8),
        ]

        blocks1d = [BNSNConv1dDBlock(64*10, mfd, 3, 1)]
        for sf in z_scale_factors:
            blocks1d.append(StridedBNSNConv1dDBlock(mfd, mfd, ks, sf))

        self.body2d = nn.Sequential(*blocks2d)
        self.body1d = nn.Sequential(*blocks1d)

        self.head = spectral_norm(nn.Conv1d(mfd, z_dim, 3, 1, 1))

    def forward(self, x):
        '''
        x.shape=(batch_size, feat_dim, num_frames)
        cond.shape=(batch_size, cond_dim, num_frames)
        '''
        bs, fd, nf = x.size()

        # ### Process generated ###
        # shape=(bs, 1, fd, nf)
        x = x.unsqueeze(1)

        # shape=(bs, 64, 10, nf_)
        x = self.body2d(x)
        # shape=(bs, 64*10, nf_)
        x = x.view(bs, -1, x.size(3))

        # ### Merging ###
        x = self.body1d(x)

        # ### Head ###
        # shape=(bs, input_size, nf)
        # out = torch.sigmoid(self.head(x))
        out = self.head(x)

        return out


if __name__ == '__main__':
    model_id = None

    if model_id is None:
        model_id = get_current_time()
        resume_training = False
    else:
        resume_training = True

    script_path = os.path.realpath(__file__)
    print(model_id)
    print(script_path)

    # Options
    base_out_dir = "./checkpoints/"

    data_dir = './training_data/exp_data'

    feat_dim = 80
    z_dim = 20

    z_scale_factors = [2, 2, 2, 2]
    z_total_scale_factor = np.prod(z_scale_factors)

    num_va = 200

    feat_type = 'mel'

    loss_funcs = OrderedDict([
        ('G', None),
        ('D', None),
        ('RealD', None),
        ('FakeD', None),
        ('Convergence', None),
    ])

    # BEGAN parameters
    gamma = 1.0
    lambda_k = 0.01
    init_k = 0.0

    # #############################################################
    # ### Set the validation losses that are used in evaluation ###
    # #############################################################
    eval_metrics = [
        ('Convergence', 'lower_better'),
    ]
    # #############################################################

    # Training options
    init_lr = 0.0001
    num_epochs = 200
    batches_per_epoch = 500

    max_grad_norm = 3

    save_rate = 10
    batch_size = 5

    # Dirs and fps
    save_dir = os.path.join(base_out_dir, 'save')
    output_dir = os.path.join(save_dir, model_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    iterator_tr, num_tr, iterator_va, _ = get_voc_datasets(
        data_dir, feat_type, batch_size, num_va)
    print('tr: {}, va: {}'.format(num_tr, num_va))

    inf_iterator_tr = make_inf_iterator(iterator_tr)

    # Prepare mean and std
    mean_fp = os.path.join(data_dir, f'mean.{feat_type}.npy')
    std_fp = os.path.join(data_dir, f'std.{feat_type}.npy')

    mean = torch.from_numpy(np.load(mean_fp)).float().cuda().view(1, feat_dim, 1)
    std = torch.from_numpy(np.load(std_fp)).float().cuda().view(1, feat_dim, 1)

    # Model
    netG = NetG(feat_dim, z_dim, z_scale_factors).cuda()
    netD = NetD(feat_dim).cuda()
    recorder = BEGANRecorder(lambda_k, init_k, gamma)

    # Optimizers
    optimizerG = optim.Adam(netG.parameters(), lr=init_lr)
    optimizerD = optim.Adam(netD.parameters(), lr=init_lr)

    # ###################################
    # ### Initialize training manager ###
    # ###################################
    manager = TrainingManager(
        [netG, netD, recorder],  # networks
        [optimizerG, optimizerD, None],  # optimizers, could be None
        ['Generator', 'Discriminator', 'BEGANRecorder'],  # names of the corresponding networks
        output_dir, save_rate, script_path=script_path)
    # ###################################

    # ### k ###
    k = recorder.k.item()

    # ### Resume training ###
    if resume_training:
        print(save_dir, model_id)
        init_epoch = manager.resume_training(model_id, save_dir)
        print(f'Resumed k: {k}')
    else:
        init_epoch = 1
        manager.save_initial()

    # ### Train ###
    for epoch in range(init_epoch, 1+num_epochs):
        print(model_id)
        t0 = time.time()

        # ### Training ###
        print('Training...')
        sum_losses_tr = OrderedDict([(loss_name, 0) for loss_name in loss_funcs])

        count_all_tr = 0

        num_batches_tr = batches_per_epoch

        tt0 = time.time()

        # In training, set net.train()
        netG.train()
        netD.train()
        for i_batch in range(batches_per_epoch):
            batch = next(inf_iterator_tr)

            # voc.shape=(bs, feat_dim, num_frames)
            voc = batch

            voc = voc.cuda()
            voc = (voc - mean) / std

            bs, _, nf = voc.size()

            # ### Train generator ###
            z = torch.zeros((bs, z_dim, int(np.ceil(nf/z_total_scale_factor)))).normal_(0, 1).float().cuda()

            fake_voc = netG(z)

            gloss = torch.mean(torch.abs(netD(fake_voc)-fake_voc))

            # Back propagation
            netG.zero_grad()
            gloss.backward(retain_graph=True)
            if max_grad_norm is not None:
                clip_grad_norm_(netG.parameters(), max_grad_norm)

            optimizerG.step()

            # ### Train discriminator ###
            real_dloss = torch.mean(torch.abs(netD(voc)-voc))
            fake_dloss = torch.mean(torch.abs(netD(fake_voc.detach())-fake_voc.detach()))

            dloss = real_dloss - k * fake_dloss

            # Update
            netD.zero_grad()
            dloss.backward()
            if max_grad_norm is not None:
                clip_grad_norm_(netD.parameters(), max_grad_norm)

            optimizerD.step()

            # ### Convergence and update k ###
            k, convergence = recorder(real_dloss, fake_dloss, update_k=True)

            # ### Losses ###
            losses = OrderedDict([
                ('G', gloss),
                ('D', dloss),
                ('RealD', real_dloss),
                ('FakeD', fake_dloss),
                ('Convergence', convergence),
            ])

            # ### Misc ###

            # Accumulate losses
            losses_tr = OrderedDict([(loss_name, lo.item()) for loss_name, lo in losses.items()])

            for loss_name, lo in losses_tr.items():
                sum_losses_tr[loss_name] += lo

            count_all_tr += 1

            # ### Print info ###
            if i_batch % 10 == 0:
                batch_info = 'Epoch {}. Batch: {}/{}, T: {:.3f}, '.format(epoch, i_batch, num_batches_tr, time.time()-tt0) + \
                    ''.join(['(tr) {}: {:.5f}, '.format(loss_name, lo) for loss_name, lo in losses_tr.items()])
                print(batch_info, k)
            tt0 = time.time()

        # Compute average loss
        mean_losses_tr = OrderedDict([(loss_name, slo / count_all_tr) for loss_name, slo in sum_losses_tr.items()])

        # ### Validation ###
        print('')
        print('Validation...')
        mean_losses_va = validate()

        # ###########################################################################
        # ### Check the best validation losses and save information in the middle ###
        # ###########################################################################

        # Check and update the best validation loss
        va_metrics = [
            (metric_name, mean_losses_va[metric_name], higher_or_lower)
            for metric_name, higher_or_lower in eval_metrics
        ]
        best_va_metrics = manager.check_best_va_metrics(va_metrics, epoch)

        # Save the record
        record = {
            'mean_losses_tr': mean_losses_tr,
            'mean_losses_va': mean_losses_va,
            'best_va_metrics': best_va_metrics,
        }

        manager.save_middle(epoch, record, va_metrics)
        manager.print_record(record)
        manager.print_record_in_one_line(best_va_metrics)
        # ###########################################################################

        t1 = time.time()
        print('Epoch: {} finished. Time: {:.3f}. Model ID: {}'.format(epoch, t1-t0, model_id))

    print(model_id)
