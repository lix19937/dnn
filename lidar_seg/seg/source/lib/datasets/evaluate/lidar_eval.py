import numpy as np
from terminaltables import AsciiTable
import torch
import os
import cv2

class Job:
    def __init__(self, iou, precision, gt, pred, path, raw_points, bev_iou) -> None:
        self.iou = iou
        self.precision = precision
        self.gt = gt
        self.pred = pred
        self.path = path
        self.raw_points = raw_points
        self.bev_iou = bev_iou
    
    def __eq__(self, other) -> bool:
        return self.bev_iou == other.bev_iou
    
    def __lt__(self, other) -> bool:
        return self.bev_iou > other.bev_iou

class BevVis(object):
    def __init__(self, ranges=(0, 100, -50, 50, -3, 3), voxel_size=(0.4, 0.4, 0.2)):
        self.ranges = ranges
        self.voxel_size = voxel_size

    def generage_bev(self):
        bev_h = (self.ranges[1] - self.ranges[0]) // self.voxel_size[0] + 1
        bev_w = (self.ranges[3] - self.ranges[2]) // self.voxel_size[1] + 1
        bev = np.ones((int(bev_h), int(bev_w), 3)).astype(np.uint8)
        return bev

    def plot_pts(self, lidar, label, p_bev, color_dict, gt_label=False):
        inds = (lidar[:, 0] > self.ranges[0]) & (lidar[:, 0] < self.ranges[1]) & \
               (lidar[:, 1] > self.ranges[2]) & (lidar[:, 1] < self.ranges[3]) & \
               (lidar[:, 2] > self.ranges[4]) & (lidar[:, 2] < self.ranges[5])

        orin_lidar_num = lidar.shape[0]
        lidar = lidar[inds]
        label = label.flatten()[:orin_lidar_num][inds]
        if gt_label:
            label[label == 255] = 0
        bev_h = (self.ranges[1] - self.ranges[0]) // self.voxel_size[0] + 1
        bev_w = (self.ranges[3] - self.ranges[2]) // self.voxel_size[1] + 1
        v = bev_h - (lidar[:, 0] - self.ranges[0]) / self.voxel_size[0]
        u = bev_w - (lidar[:, 1] - self.ranges[2]) / self.voxel_size[1]
        v = np.clip(v, 0, bev_h - 1)
        u = np.clip(u, 0, bev_w - 1)
        v = v.reshape(-1).astype(np.int)
        u = u.reshape(-1).astype(np.int)
        order = np.argsort(lidar[:, 2])
        p_bev[v[order], u[order], ] = [color_dict[i][::-1] for i in label[order]]
        # if kitti: color_dict[i & 0xFF]
        return p_bev

    def get_bev_index(self, lidar):
        C = lidar.shape[0]
        lidar = lidar.reshape(C, -1)
        bev_h = (self.ranges[1] - self.ranges[0]) // self.voxel_size[0] + 1
        bev_w = (self.ranges[3] - self.ranges[2]) // self.voxel_size[1] + 1
        v = bev_h - (lidar[0, :] - self.ranges[0]) / self.voxel_size[0]
        u = bev_w - (lidar[1, :] - self.ranges[2]) / self.voxel_size[1]
        v = np.clip(v, 0, bev_h - 1)
        u = np.clip(u, 0, bev_w - 1)
        v = v.reshape(-1).astype(np.int)
        u = u.reshape(-1).astype(np.int)
        bev_idx = v * int(bev_w) + u
        order = np.argsort(lidar[2, :])
        p_bev = np.ones((int(bev_h) * int(bev_w),), np.int32) * -1
        p_bev[bev_idx[order]] = order
        b_idx = p_bev[p_bev > 0]
        # if kitti: color_dict[i & 0xFF]
        return b_idx


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    hist = bin_count[:n ** 2].reshape(n, n)
    # do not infer suspension
    hist[3, :] = 0
    hist[:, 3] = 0
    return hist


def per_class_status(hist):
    tp = np.diag(hist)
    fp = hist.sum(0) - tp
    fn = hist.sum(1) - tp
    tp_sum = tp.sum()
    fp_sum = fp.sum()
    m_precision = tp_sum / (tp_sum + fp_sum)
    m_iou = tp_sum / (tp_sum + fp_sum + fp_sum)
    np.seterr(divide='ignore', invalid='ignore')
    p_iou = tp / (fp + fn + tp)
    p_precision = tp / (tp + fp)
    p_recall = tp / (tp + fn)
    # m_iou = m_iou if np.isnan(p_iou[-1]) else p_iou[-1]
    return p_iou, p_precision, p_recall, m_iou, m_precision


def fast_hist_crop(pd, gt, unique_label):
    hist = fast_hist(pd.flatten(), gt.flatten(), np.max(unique_label))
    # do not infer suspension
    # hist[3, :] = 0
    # hist[:, 3] = 0
    return hist


def get_acc(hist):
    """Compute the overall accuracy.

    Args:
        hist(np.ndarray):  Overall confusion martix
        (num_classes, num_classes ).

    Returns:
        float: Calculated overall acc
    """

    return np.diag(hist).sum() / hist.sum()


def get_acc_cls(hist):
    """Compute the class average accuracy.

    Args:
        hist(np.ndarray):  Overall confusion martix
        (num_classes, num_classes ).

    Returns:
        float: Calculated class average acc
    """

    return np.nanmean(np.diag(hist) / hist.sum(axis=1))


def get_bad_instance(mean_status, k=3, metrics=['iou', 'precision', 'recall']):
    metric_id = []
    for metric in metrics:
        if metric == 'iou':
            metric_id.append(0)
        elif metric == 'precision':
            metric_id.append(1)
        elif metric == 'recall':
            metric_id.append(2)
    # mean_status = np.nanmean(total_status[:, metric_id, :], axis=2)
    idx = np.argsort(mean_status, axis=0)[:k, :]
    ret = {}
    for i, metric in enumerate(metrics):
        ret[metric + '_worst'] = [idx[:, i].tolist(), mean_status[idx[:, i], i].tolist()]
        ret['mean_' + metric] = np.mean(mean_status, axis=0)[i]
    return ret


def show_worst_instance(raw_cloud, pred, gt, pcd_path, palette, root_dir):
    file_name = pcd_path.split('/')[-1][:-4]
    pred_img = bev(raw_cloud, pred, palette, file_name + '_pred', gt_label=False, save_dir=root_dir)
    gt_img = bev(raw_cloud, gt, palette, file_name + '_gt', gt_label=True, save_dir=root_dir)
    return pred_img, gt_img


def bev(lidar, label, palette, file_name, gt_label, save_dir):
    lidar_vis = BevVis()
    temp_bev = lidar_vis.plot_pts(lidar, label, lidar_vis.generage_bev(), palette, gt_label=gt_label)
    img_path = os.path.join(os.path.realpath(save_dir), 'hardcase_image', f'{file_name}.jpg')
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    if not os.path.exists(img_path):
        print(img_path)
    cv2.imwrite(img_path, cv2.resize(temp_bev, (1000, 1000)))
    return img_path

def calc_hist(result, indices, ignore_index, hists, status):
    gt_label = result['gt_label']
    seg_pred = result['predict_labels']
    if isinstance(gt_label, torch.Tensor):
        gt_seg = gt_label.clone().numpy().astype(np.int).flatten()
    else:
        gt_seg = gt_label.flatten()
    if isinstance(seg_pred, torch.Tensor):
        pred_seg = seg_pred.clone().numpy().astype(np.int).flatten()
    else:
        pred_seg = seg_pred.flatten()
    # filter out ignored points
    pred_seg[gt_seg == ignore_index] = -1
    gt_seg[gt_seg == ignore_index] = -1
    level_1 = indices[0].flatten()
    level_2 = indices[1].flatten()
    level_3 = indices[2].flatten()
    # calculate one instance result
    hist_per_frame = fast_hist(pred_seg, gt_seg, ignore_index)
    hist_per_frame_l1 = fast_hist(pred_seg[level_1], gt_seg[level_1], ignore_index)
    hist_per_frame_l2 = fast_hist(pred_seg[level_2], gt_seg[level_2], ignore_index)
    hist_per_frame_l3 = fast_hist(pred_seg[level_3], gt_seg[level_3], ignore_index)

    bev = BevVis()
    b_idx = bev.get_bev_index(result['raw_points'])
    hist_per_frame_bev = fast_hist(pred_seg[b_idx], gt_seg[b_idx], ignore_index)

    status.put(Job(*per_class_status(hist_per_frame)[3:], gt_seg, pred_seg,
    result['path'], result['raw_points'], per_class_status(hist_per_frame_bev)[3]))
    if (len(status.queue) > 100):
        a = status.get()
        status.task_done()
    hists[0] += hist_per_frame
    hists[1] += hist_per_frame_l1
    hists[2] += hist_per_frame_l2
    hists[3] += hist_per_frame_l3


def lidar_eval(total_hist, label2cat, ignore_index):
    """Semantic Segmentation  Evaluation.

    Evaluate the result of the Semantic Segmentation.

    Args:
        gt_labels (list[torch.Tensor]): Ground truth labels.
        seg_preds  (list[torch.Tensor]): Predictions.
        label2cat (dict): Map from label to category name.
        ignore_index (int): Index that will be ignored in evaluation.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.

    Returns:
        dict[str, float]: Dict of results.
    """
        
    iou, precision, recall, m_iou, m_precision = per_class_status(total_hist[0])
    iou_l1, precision_l1, recall_l1, miou_l1, mprecision_l1 = per_class_status(total_hist[1])
    iou_l2, precision_l2, recall_l2, miou_l2, mprecision_l2 = per_class_status(total_hist[2])
    iou_l3, precision_l3, recall_l3, miou_l3, mprecision_l3 = per_class_status(total_hist[3])

    header_list = [[iou, precision, recall],
                   [iou_l1, precision_l1, recall_l1],
                   [iou_l2, precision_l2, recall_l2],
                   [iou_l3, precision_l3, recall_l3]]

    m_recall = m_precision

    # acc = get_acc(total_hist[0])
    # acc_cls = get_acc_cls(total_hist[0])
    # ret_dict = dict()
    # ret_dict['mean_iou'] = float(m_iou)
    # ret_dict['mean_precision'] = float(m_precision)
    # ret_dict['mean_recall'] = float(m_recall)
    # ret_dict['acc'] = float(acc)
    # ret_dict['acc_cls'] = float(acc_cls)

    header = ['Classes', 'iou', 'precision', 'recall']
    t_columns = []
    for i in range(len(label2cat)):
        if i == ignore_index:
            continue
        # ret_dict[label2cat[i]] = float(iou[i])
        for j, dist in enumerate(['', '[0,30]', '[30,50]', '[50,]']):
            t_c = [[label2cat[i] + dist]]
            for h in header_list[j]:
                t_c.append([f'{h[i]:.3f}'])
            t_columns.append(t_c)

    t_columns.append([['mean[0,30]'], [f'{miou_l1:.3f}'], [f'{mprecision_l1:.3f}'], [f'{mprecision_l1:.3f}']])
    t_columns.append([['mean[30,50]'], [f'{miou_l2:.3f}'], [f'{mprecision_l2:.3f}'], [f'{mprecision_l2:.3f}']])
    t_columns.append([['mean[50,]'], [f'{miou_l3:.3f}'], [f'{mprecision_l3:.3f}'], [f'{mprecision_l3:.3f}']])
    t_columns.append([['mean'], [f'{m_iou:.3f}'], [f'{m_precision:.3f}'], [f'{m_recall:.3f}']])

    table_data = [header]
    table_rows = [list(zip(*t_col)) for t_col in t_columns]
    for row in table_rows:
        table_data += row
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    # print_log('\n' + table.table, logger=logger)
    return table, m_iou
