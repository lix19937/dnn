import torch
import torch.nn as nn

PI = 3.141592653589793


class KLDloss(nn.Module):
    def __init__(self, taf=1.0, reduction="none", loss_weight=1.0):
        super(KLDloss, self).__init__()
        self.reduction = reduction
        self.taf = taf
        self.loss_weight = loss_weight

    def forward(self, pred, target):  # pred [[x,y,w,h,angle], ...]
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 5)
        target = target.view(-1, 5)

        delta_x = pred[:, 0] - target[:, 0]
        delta_y = pred[:, 1] - target[:, 1]
        pre_angle_radian = pred[:, 4]
        target_angle_radian = target[:, 4]
        delta_angle_radian = pre_angle_radian - target_angle_radian

        kld = 0.5 * (4 * torch.pow((delta_x.mul(torch.cos(target_angle_radian)) + delta_y.mul(torch.sin(target_angle_radian))), 2) / torch.pow(target[:, 2], 2)
                     + 4 * torch.pow((delta_y.mul(torch.cos(target_angle_radian)) - delta_x.mul(torch.sin(target_angle_radian))), 2) / torch.pow(target[:, 3], 2)) \
              + 0.5 * (
                      torch.pow(pred[:, 3], 2) / torch.pow(target[:, 2], 2) * torch.pow(torch.sin(delta_angle_radian), 2)
                      + torch.pow(pred[:, 2], 2) / torch.pow(target[:, 3], 2) * torch.pow(torch.sin(delta_angle_radian), 2)
                      + torch.pow(pred[:, 3], 2) / torch.pow(target[:, 3], 2) * torch.pow(torch.cos(delta_angle_radian), 2)
                      + torch.pow(pred[:, 2], 2) / torch.pow(target[:, 2], 2) * torch.pow(torch.cos(delta_angle_radian), 2)) \
              + 0.5 * (torch.log(torch.pow(target[:, 3], 2) / torch.pow(pred[:, 3], 2))
                       + torch.log(torch.pow(target[:, 2], 2) / torch.pow(pred[:, 2], 2))
                       ) - 1.0

        kld_loss = 1 - 1 / (self.taf + torch.log(kld + 1))

        return kld_loss * self.loss_weight


def compute_kld_loss(targets, preds):
    with torch.no_grad():
        kld_loss_ts_ps = torch.zeros(0, preds.shape[0], device=targets.device)
        for target in targets:
            target = target.unsqueeze(0).repeat(preds.shape[0], 1)
            kld_loss_t_p = kld_loss(preds, target)
            kld_loss_ts_ps = torch.cat((kld_loss_ts_ps, kld_loss_t_p.unsqueeze(0)), dim=0)
    return kld_loss_ts_ps


def kld_loss(pred, target, taf=1.0):  # pred [[x,y,w,h,angle], ...]
    assert pred.shape[0] == target.shape[0]

    pred = pred.view(-1, 5)
    target = target.view(-1, 5)

    delta_x = pred[:, 0] - target[:, 0]
    delta_y = pred[:, 1] - target[:, 1]
    #
    # pre_angle_radian = 2 * PI * (pred[:, 4] - 0.5)
    # target_angle_radian = 2 * PI * (target[:, 4] - 0.5)
    pre_angle_radian = pred[:, 4]
    target_angle_radian = target[:, 4]
    delta_angle_radian = pre_angle_radian - target_angle_radian

    kld = 0.5 * (4 * torch.pow((delta_x.mul(torch.cos(target_angle_radian)) + delta_y.mul(torch.sin(target_angle_radian))), 2) / torch.pow(target[:, 2], 2)
                 + 4 * torch.pow((delta_y.mul(torch.cos(target_angle_radian)) - delta_x.mul(torch.sin(target_angle_radian))), 2) / torch.pow(target[:, 3], 2)) \
          + 0.5 * (
                  torch.pow(pred[:, 3], 2) / torch.pow(target[:, 2], 2) * torch.pow(torch.sin(delta_angle_radian), 2)
                  + torch.pow(pred[:, 2], 2) / torch.pow(target[:, 3], 2) * torch.pow(torch.sin(delta_angle_radian), 2)
                  + torch.pow(pred[:, 3], 2) / torch.pow(target[:, 3], 2) * torch.pow(torch.cos(delta_angle_radian), 2)
                  + torch.pow(pred[:, 2], 2) / torch.pow(target[:, 2], 2) * torch.pow(torch.cos(delta_angle_radian), 2)
          ) \
          + 0.5 * (
                  torch.log(torch.pow(target[:, 3], 2) / torch.pow(pred[:, 3], 2))
                  + torch.log(torch.pow(target[:, 2], 2) / torch.pow(pred[:, 2], 2))
          ) \
          - 1.0

    kld_loss = 1 - 1 / (taf + torch.log(kld + 1))

    return kld_loss

# loss = KLDloss()
# pred = torch.tensor([[20, 20, 10, 10, -90], [20, 20, 20, 10, 90], [1, 0.5, 2, 1, 0]], dtype=torch.float32)
# target = torch.tensor([[20, 20, 10, 10, -90], [20, 20, 20, 10, 0], [0.5, 1, 2, 1, -90]], dtype=torch.float32)
# kld = kld_loss(pred, target)
# print(kld)


# pred = torch.tensor([[20, 20, 10, 10, -90], [20, 20, 20, 10, 90], [1, 0.5, 2, 1, 0]], dtype=torch.float32)
# target = torch.tensor([[20, 20, 10, 10, -90], [20, 20, 20, 10, 0]], dtype=torch.float32)
# kld = compute_kld_loss(target, pred)
# print(kld)
#
# print(torch.floor(torch.tensor(-9.9)))
