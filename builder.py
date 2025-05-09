# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
import time

import torchvision
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from resnet import ResNet18
from einops import rearrange
from .region_focused import RegionFocused

class RFMaCo(nn.Module):
    """
    Build a RFMaCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(RFMaCo, self).__init__()

        self.T = T
        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        self.rfmodel = ResNet18()
        state_dict = torch.load('./model_state_dict_resnet18.pt')

        self.rfmodel.load_state_dict(state_dict)
        self.rf = RegionFocused(lr=0.1, p=1, gamma=0.01, beta=10.0, num_steps=100)
        #self.rf = RegionFocused(lr=0.1, p=1, gamma=0.0005, beta=500.0, num_steps=100)

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                #mlp.append(nn.LayerNorm(dim2))
                mlp.append(nn.ReLU(inplace=True))
                #mlp.append(nn.GELU())
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))
                #mlp.append(nn.LayerNorm(dim2,elementwise_affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def generate_mask(self, masks, mask_ratio=0.25, images=None, guide=True, epoch=0, total_epoch=300):
        masks = F.interpolate(masks, size=(224, 224), mode='bicubic', align_corners=False)
        images = F.interpolate(images, size=(224, 224), mode='bicubic', align_corners=False).to(images.device)
        batch_size,channels,height,width = masks.shape
        patch_size = 16
        masks_patches = rearrange(masks, 'b c (h p1) (w p2) -> b (h w) c p1 p2', p1=patch_size, p2=patch_size)

        masks_patches_mean = masks_patches.mean(dim=[2, 3, 4])  
        N, L= masks_patches_mean.shape  
        len_keep = int(L * (1 - mask_ratio))
        ids_shuffle_loss = torch.argsort(masks_patches_mean, dim=1)  # (N, L)

        keep_ratio = 0.5
        ids_shuffle = torch.zeros_like(ids_shuffle_loss, device=masks.device).int()

        if guide:
            keep_ratio = float((epoch + 1) / total_epoch) * 0.5

        if int((L - len_keep) * keep_ratio) <= 0:
            noise = torch.randn(N, L, device=masks.device)
            ids_shuffle = torch.argsort(noise, dim=1)
        else:
            for i in range(N):
                len_loss = int((L - len_keep) * keep_ratio)
                ids_shuffle[i, -len_loss:] = ids_shuffle_loss[i, -len_loss:]

                temp = torch.arange(L, device=masks.device)
                deleted = np.delete(temp.cpu().numpy(), ids_shuffle[i, -len_loss:].cpu().numpy())
                np.random.shuffle(deleted)
                ids_shuffle[i, :(L - len_loss)] = torch.tensor(deleted, device=masks.device, dtype=torch.int64)

        ids_restore = torch.argsort(ids_shuffle, dim=1)

        masks = torch.zeros([N, L], device=masks.device)
        masks[:, :len_keep] = 1
        masks = torch.gather(masks, dim=1, index=ids_restore)

        return images, masks

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        logits = logits.cuda()
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        #labels = torch.arange(N, dtype=torch.long).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def byol_loss(self, q, k):
        q = nn.functional.normalize(q, dim=-1, p=2)
        k = nn.functional.normalize(k, dim=-1, p=2)
        return 2 - 2 * (q * k).sum(dim=-1).mean()

    def forward(self, x1, x2, m, epoch, args):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: RFMaCo momentum
        Output:
            loss
        """
        self.rfmodel.eval()
        masks_x1 = self.rf(self.rfmodel, x1).to(x1.device)
        x1, masks = self.generate_mask(masks_x1, images=x1, epoch=epoch, total_epoch=args.epochs)

        q1 = self.base_encoder(x1, masks)
        q1 = self.predictor(q1)

        q2 = self.base_encoder(x2)
        q2 = self.predictor(q2)

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1, masks)
            k2 = self.momentum_encoder(x2)

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
        #return self.byol_loss(q1, k2) + self.byol_loss(q2, k1)


class RFMaCo_ViT(RFMaCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
