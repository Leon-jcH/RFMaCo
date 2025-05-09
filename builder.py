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

from issba_resnet import ResNet18_100
from resnet import ResNet18
from einops import rearrange
#from .region_focused import CognitiveDistillation
from .cognitive_distillation import CognitiveDistillation

class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T
        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        self.cdmodel = ResNet18()
        state_dict = torch.load('/mnt/data/hejiacheng/pythonproject/CognitiveDistillation-main/result_Aug/BadNetCIFAR10_rn18/checkpoints/model_state_dict_resnet18.pt')
        #self.cdmodel = ResNet18_100()
        #state_dict = torch.load('/mnt/data/hejiacheng/pythonproject/CognitiveDistillation-main/result/BadNetImageNet_rn18/checkpoints/model_state_dict_resnet18.pt')

        self.cdmodel.load_state_dict(state_dict)
        self.cd = CognitiveDistillation(lr=0.1, p=1, gamma=0.01, beta=10.0, num_steps=100)
        #self.cd = CognitiveDistillation(lr=0.1, p=1, gamma=0.0005, beta=500.0, num_steps=100)

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
        # 调整掩码和图像大小
        masks = F.interpolate(masks, size=(224, 224), mode='bicubic', align_corners=False)
        images = F.interpolate(images, size=(224, 224), mode='bicubic', align_corners=False).to(images.device)
        batch_size,channels,height,width = masks.shape
        # 将掩码拆分为 16x16 的 patches
        patch_size = 16
        masks_patches = rearrange(masks, 'b c (h p1) (w p2) -> b (h w) c p1 p2', p1=patch_size, p2=patch_size)

        # 计算每个 patch 的像素均值
        masks_patches_mean = masks_patches.mean(dim=[2, 3, 4])  # 形状为 [batch_size, num_patches]
        N, L= masks_patches_mean.shape  # 获取批量大小和patch数量
        len_keep = int(L * (1 - mask_ratio))
        # 根据 patch 的均值进行排序
        ids_shuffle_loss = torch.argsort(masks_patches_mean, dim=1)  # (N, L)

        # 初始化 ids_shuffle
        keep_ratio = 0.5
        ids_shuffle = torch.zeros_like(ids_shuffle_loss, device=masks.device).int()

        # 根据 epoch 变化调整 keep_ratio
        if guide:
            keep_ratio = float((epoch + 1) / total_epoch) * 0.5

        # 判断需要保留的掩码数量，如果不需要保留，则随机打乱
        if int((L - len_keep) * keep_ratio) <= 0:
            # 随机打乱
            noise = torch.randn(N, L, device=masks.device)
            ids_shuffle = torch.argsort(noise, dim=1)
        else:
            for i in range(N):
                # 保留 top keep_ratio 的掩码，并随机选择剩余的掩码
                len_loss = int((L - len_keep) * keep_ratio)
                #将 ids_shuffle_loss 中排序靠前的 len_loss 个掩码保留到 ids_shuffle 中对应的位置
                ids_shuffle[i, -len_loss:] = ids_shuffle_loss[i, -len_loss:]

                # 随机选择剩余的掩码
                temp = torch.arange(L, device=masks.device)
                deleted = np.delete(temp.cpu().numpy(), ids_shuffle[i, -len_loss:].cpu().numpy())
                np.random.shuffle(deleted)
                ids_shuffle[i, :(L - len_loss)] = torch.tensor(deleted, device=masks.device, dtype=torch.int64)

        # 生成 ids_restore 用于恢复顺序
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # 生成掩码：0 表示keep，1 表示remove
        masks = torch.zeros([N, L], device=masks.device)
        masks[:, :len_keep] = 1
        # 恢复顺序以得到最终掩码
        masks = torch.gather(masks, dim=1, index=ids_restore)
        # 将掩码扩展回原始形状
        '''masks_vis = masks.view(batch_size, -1, 1, 1).expand(-1, -1, patch_size,
                                                      patch_size)  # [batch_size, num_patches, p1, p2]
        #masks_vis = rearrange(masks_vis, 'b (h w) p1 p2 -> b 1 (h p1) (w p2)', h=height // patch_size, w=width // patch_size,
                         p1=patch_size, p2=patch_size)
        # 将掩码应用到图像
        images1 = images * masks_vis'''

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
            m: moco momentum
        Output:
            loss
        """
        self.cdmodel.eval()
        masks_x1 = self.cd(self.cdmodel, x1).to(x1.device)
        x1, masks = self.generate_mask(masks_x1, images=x1, epoch=epoch, total_epoch=args.epochs)
        #imshow(torchvision.utils.make_grid(x1), "x1")
        #imshow(torchvision.utils.make_grid(masks_x1), "masks")
        #print(masks)
        #imshow(torchvision.utils.make_grid(masks_vis), "masks_vis")
        #imshow(torchvision.utils.make_grid(images1), "images1")
        # compute features
        #q1, _ = self.base_encoder(x1, mask=masks)
        q1 = self.base_encoder(x1, masks)
        q1 = self.predictor(q1)
        #print(x1.shape)

        #q2, _ = self.base_encoder(x2)
        q2 = self.base_encoder(x2)
        q2 = self.predictor(q2)
        #print(x2.shape)

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            #k1, _ = self.momentum_encoder(x1, mask=masks)
            k1 = self.momentum_encoder(x1, masks)
            k2 = self.momentum_encoder(x2)

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
        #return self.byol_loss(q1, k2) + self.byol_loss(q2, k1)


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


def imshow(img, title):
    img = img / 2 + 0.5  # 非标准化
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

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