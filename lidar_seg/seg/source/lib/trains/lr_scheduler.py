import matplotlib.pyplot as plt
import numpy as np


def update_lr(epoch, warmup_epochs, warmup_start_lr, total_epochs, lr0, expo):
    warmup_factor = (lr0 / warmup_start_lr) ** (1/warmup_epochs)
    if epoch < warmup_epochs:
        lr = warmup_start_lr * (warmup_factor ** epoch)
    else:
        lr = np.power(1.0 - (epoch - warmup_epochs) /
                      (total_epochs - warmup_epochs), expo) * lr0
    return lr


def finetune_update_lr(epoch, cur_lr):
    lrschedule = {
        0: 5*1e-6,
        10: 1e-6,
        30: 1e-6
    }

    if epoch in lrschedule:
        lr = lrschedule[epoch]
    else:
        lr = cur_lr    
    return lr


def update_lr_new(epoch, warmup_epochs, num_epochs, warmup_start_lr, lr0):
    warmup_factor = (lr0 / warmup_start_lr) ** (1.1/warmup_epochs)
    if epoch < warmup_epochs:
        lr = warmup_start_lr * (warmup_factor ** epoch)
    else:
        lr = (lr0 * pow(1 - (epoch / num_epochs), 0.95) -
              0.01 * lr0 * pow(epoch // 10 + 1, 1.3))
        lr = max(lr, lr0 / 100.0)
    return lr



def vis_lr_curve():
    warmup_start_lr = 1e-5
    warmup = 5
    total = 200
    lr0 = 1e-3
    expo = 0.8
    x = [i for i in range(total)]
    y = [update_lr(i, warmup, warmup_start_lr, total, lr0, expo) for i in x]
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    vis_lr_curve()
