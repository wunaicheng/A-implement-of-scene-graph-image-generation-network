import torch
import torch.nn.functional as F


def model_loss(imgs_true, imgs_pred, boxes_true, boxes_pred, masks_true, masks_pred, pixel_weight=1, box_weight=10, mask_weight=0.1):
    pixel_loss = F.l1_loss(imgs_pred, imgs_true)
    boxes_loss = F.mse_loss(boxes_pred, boxes_true)
    masks_loss = F.binary_cross_entropy(masks_pred, masks_true)
    return pixel_weight*pixel_loss + box_weight*boxes_loss + mask_weight*masks_loss


def bce_loss(inp, target):
    neg_abs = -inp.abs()
    loss = inp.clamp(min=0) - inp * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def make_targets(x, y):
    return torch.full_like(x, y)


def gan_gen_loss(scores_fake, discriminator_weight=0.01):
    if scores_fake.dim() > 1:
        scores_fake = scores_fake.view(-1)
    y_fake = make_targets(scores_fake, 1)
    return discriminator_weight*bce_loss(scores_fake, y_fake)


def gan_dis_loss(scores_real, scores_fake):
    if scores_real.dim() > 1:
        scores_real = scores_real.view(-1)
        scores_fake = scores_fake.view(-1)
    y_real = make_targets(scores_real, 1)
    y_fake = make_targets(scores_fake, 0)
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real + loss_fake
