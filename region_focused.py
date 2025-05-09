import torch
import torch.nn as nn
import torch.optim as optim

def smoothness_loss(img, weight=1):
    smooth_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).mean(dim=[1, 2, 3])
    smooth_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).mean(dim=[1, 2, 3])
    return weight * (smooth_h + smooth_w)


class RegionawareFocused(nn.Module):
    def __init__(self, lr=0.1, p=1, gamma=0.01, beta=1.0, num_steps=100, mask_channel=1, norm_only=False):
        super(CognitiveDistillation, self).__init__()
        self.p = p
        self.gamma = gamma
        self.beta = beta
        self.num_steps = num_steps
        self.lr = lr
        self.mask_channel = mask_channel
        self.norm_only = norm_only
        self.l1 = nn.L1Loss(reduction='none')
        self._EPSILON = 1.e-6

    def get_raw_mask(self, mask):
        return (torch.tanh(mask) + 1) / 2

    def forward(self, model, images, labels=None):
        model.eval()
        b, c, h, w = images.shape
        mask = torch.ones(b, self.mask_channel, h, w, device=images.device)
        mask_param = nn.Parameter(mask)
        optimizer = optim.Adam([mask_param], lr=self.lr, betas=(0.9, 0.999))

        with torch.no_grad():
            logits = model(images)

        for step in range(self.num_steps):
            optimizer.zero_grad()
            mask = self.get_raw_mask(mask_param) 
            x_adv = images * mask + (1 - mask) * torch.rand(b, c, h, w, device=images.device)

            adv_logits = model(x_adv)
            loss = self.l1(adv_logits, logits).mean(dim=1)

            norm = torch.norm(mask, p=self.p, dim=[1, 2, 3]) * self.gamma
            s_l = smoothness_loss(mask, weight=self.beta)

            loss_total = loss + norm + s_l
            loss_total.mean().backward()
            optimizer.step()

        mask = self.get_raw_mask(mask_param).detach().cpu()
        if self.norm_only:
            return torch.norm(mask, p=1, dim=[1, 2, 3])
        return mask
