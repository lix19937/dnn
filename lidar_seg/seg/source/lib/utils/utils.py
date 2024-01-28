from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


def filename_compare(x, y):
    name_x = x.split('/')[-1]
    name_y = y.split('/')[-1]
    assert 'bin' in name_x or 'json' in name_x or 'txt' in name_x
    if 'bin' in name_x or 'txt' in name_x:
        time_x = name_x[:-4]
        time_y = name_y[:-4]
    elif 'json' in name_x:
        time_x = name_x[:-5]
        time_y = name_y[:-5]
    if time_x < time_y:
        return 1
    elif time_x > time_y:
        return -1
    else:
        return 0


def check_timestamp(clouds, labels):
    """1. timestamp of clouds must be equal to labels.
          2. timestamps must be unique.
    """
    time_set = set()
    for idx in range(len(clouds)):
        time_cloud = clouds[idx].split('/')[-1].replace('.bin', '')
        time_label = labels[idx].split(
            '/')[-1].replace('.json', '').replace('.txt', '')
        assert time_cloud == time_label
        time_set.add(time_cloud)
    assert len(time_set) == len(clouds)


def write_results(objects, file, score_thr=0.5):
    obj_json = []
    # x,y,z,l,w,h,theta,score,cls.
    for obj in objects:
        if obj[-2] < score_thr:  # score threshold.
            continue
        obj_json.append({
            'x': obj[0].tolist(),
            'y': obj[1].tolist(),
            'z': obj[2].tolist(),
            'l': obj[3].tolist(),
            'w': obj[4].tolist(),
            'h': obj[5].tolist(),
            'theta': obj[6].tolist(),
            'score': obj[7].tolist(),
            'cls': obj[8].tolist(),
        })
    with open(file, 'w') as f:
        json.dump(obj_json, f)
