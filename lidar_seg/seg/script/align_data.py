import os
import os.path as osp
import shutil
from util import is_float

table = {
        'label_20220417':['city', 'sunny'],
        'label_20220429':['city', 'sunny'],
        'label_20220507':['city', 'sunny'],
        'label_20220510':['city', 'sunny'],
        'label_20220520':['park', 'sunny'], 
        'label_20220523':['park', 'rainy'],
        'label_20220530':['park', 'sunny'],
        'label_20220614':['elevated', 'sunny'],
        'label_20220621':['city', 'sunny'],
        'label_20220623':['city', 'sunny'],
        'label_20220628':['city', 'sunny'],
        'label_20220701':['city', 'sunny'],
        'label_20220704':['elevated', 'rainy'],
        'label_20220713':['city', 'sunny'],
        'label_20220719':['city', 'sunny'],
        'label_20220726':['city', 'sunny'],
        'label_20220803':['city', 'sunny'],
        'label_20220809':['city', 'rainy'],
        'label_20220817':['open', 'sunny'],
        'label_20220823':['cone_baffle', 'sunny'],
        'label_20220830':['city', 'sunny'],
        'label_20220906':['city', 'sunny'],
        'label_20220913':['open', 'rainy'],
        'label_20220919':['trunk', 'sunny'],
        'label_20220926':['cross', 'sunny'],
}

if __name__ == '__main__':

    origin_path = "/home/igs/mnt/lidar_label/数据标注/样例_障碍物/项目交付/LidarSEG/all_unzip/11月交付/11.6-11.10"
    task_idx = -1

    target_path = "/data/luminar_seg/single_seg_1111"
    img_dir = os.path.join(target_path, 'pcd_dir')
    img_list1 = os.listdir(img_dir)
    img_list = [_[:20] for _ in img_list1]
    img_dict = dict(zip(img_list, img_list1))
    filter_list = open(os.path.join(target_path, 'filter.txt'), 'r').readlines()
    filter_list = [_.strip().split(' ')[0][:20] for _ in filter_list]
    severe = open(os.path.join(target_path, 'filter_severe.txt'), 'r').readlines()
    severe = [_.strip().split(' ')[0][:20] for _ in severe]
    severe = set(severe)
    print('severe length is ', len(severe))
    img_set = set(img_list)
    filter_set = set(filter_list)
    img_set.difference_update(filter_set)
    origin_task = os.listdir(origin_path)
    for data in origin_task:
        tasks = osp.join(origin_path, data)
        for root, dirs, _ in os.walk(tasks):
            if len(dirs) > 0 and 'label_' in dirs[0]:
                break
        for task in os.listdir(root):
            task_dir = os.path.join(root, task)
            for troot, tdirs, _ in os.walk(task_dir):
                if len(tdirs) > 3 and is_float(tdirs[0][:-4]):
                    break
            ttroot = osp.dirname(troot)
            ttdir = os.listdir(ttroot)
            for tpath in ttdir:
                final_root = osp.join(ttroot, tpath)
                files = os.listdir(final_root)
                tgt_set = set(files)
                same_list = list(img_set.intersection(tgt_set))
                severe_same = list(tgt_set.intersection(severe))
                origin_list = origin_path.split('/')
                tgt_dir = '/'.join(origin_list[:(task_idx - 2)]) + f'/wrong_single/{origin_list[task_idx]}/{task}'
                if len(same_list) > 0:
                    if len(severe_same) > 0:
                        os.makedirs(tgt_dir, exist_ok=True)
                        [shutil.copytree(osp.join(final_root, _), osp.join(tgt_dir, _)) for _ in severe_same]
                    t = table[task]
                    same_list.sort()
                    print(len(same_list))
                    os.makedirs(f'/data/luminar_seg/aligned_seg/single/{t[1]}/{t[0]}', exist_ok=True)
                    f = open(f'/data/luminar_seg/aligned_seg/single/{t[1]}/{t[0]}/train.txt', 'a')
                    [f.write(os.path.join(img_dir, img_dict[_]) + '\n') for _ in same_list]
                    f.close()
    