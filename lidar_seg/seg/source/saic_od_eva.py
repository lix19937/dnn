import gc
import json
import math
import numba
import numpy as np
from numba import cuda
import cv2
import sys
import os


def saic_od_eval(gt_annos, dt_annos, current_classes, distance):
    eval_types = ['3d']
    overlap_0_7 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5,0.5,0.5,0.5,0.5,0.5,0.5])
    min_overlaps = np.stack([overlap_0_7], axis=0)  # [2, 3, 5]
    class_to_name = {0:'Car', 1:'Van', 2:'Bus', 3:'Truck', 4:'Head', 5:'Tricycle', 6:'Vehicle_part',
                          7: 'Dynamic_other', 8: 'Pedestrian', 9: 'Bike', 10: 'Cyclist', 11: 'Animal'}
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):  # to list
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, current_classes]
    compute_aos = False
    pred_alpha = True
    valid_alpha_gt = True
    compute_aos = (pred_alpha and valid_alpha_gt)
    if compute_aos:
        eval_types.append('aos')
    ret = do_eval(gt_annos, dt_annos, current_classes, min_overlaps, distance, eval_types)
    return  ret


def do_eval(gt_annos,
            dt_annos,
            current_classes,
            min_overlaps,
            distance,
            eval_types=['3d']):
    difficultys = [0]
    ret = eval_class(
        gt_annos,
        dt_annos,
        current_classes,
        distance,
        2,
        min_overlaps,
        compute_aos=('aos' in eval_types))
    return ret

def eval_class(gt_annos,
               dt_annos,
               current_classes,
               distance,
               metric,
               min_overlaps,
               compute_aos=False,
               num_parts=200):

    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    if num_examples < num_parts:
        num_parts = num_examples
    split_parts = get_split_parts(num_examples, num_parts)  # 多了分拆

    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    N_SAMPLE_PTS = 1
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    num_distance = len(distance)
    precision = np.zeros(
        [num_class, num_distance])
    recall = np.zeros(
        [num_class, num_distance])
    deviation_list = []
    gt_num_list = []
    dt_num_list = []
    tp_num_list = np.zeros(
        [num_class, num_distance])
    fp_num_list = np.zeros(
        [num_class, num_distance])
    fn_num_list = np.zeros(
        [num_class, num_distance])
    aos = np.zeros([num_class, num_distance, num_minoverlap, N_SAMPLE_PTS])
    for m, current_class in enumerate(current_classes):
        d_list = []
        g_num = []
        d_num =[]
        for idx_l, dis in enumerate(distance):
            rets = _prepare_data(gt_annos, dt_annos, current_class, dis)
            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, gt_data_detail_list,
             dt_data_detail_list, dontcares, total_dc_num, total_num_valid_gt, total_dt_num) = rets
            min_overlap = min_overlaps[0][0]
            rets = compute_statistics_jit(
                overlaps[0],
                gt_datas_list[0],
                dt_datas_list[0],
                ignored_gts[0],
                ignored_dets[0],
                gt_data_detail_list[0],
                dt_data_detail_list[0],
                min_overlap=min_overlap,
                thresh=0.5,
                compute_fp=True)
            tp, fp, fn, similarity, thresholds, list_deviation= rets
            d_list.append(list_deviation)
            g_num.append(total_num_valid_gt)
            d_num.append(total_dt_num)
            tp_num_list[current_class][idx_l] = tp
            fp_num_list[current_class][idx_l] = fp
            fn_num_list[current_class][idx_l] = fn
            if(tp == 0):
                recall[current_class][idx_l] = 0
                precision[current_class][idx_l] = 0
            else:
                recall[current_class][idx_l] = tp / (tp + fn)
                precision[current_class][idx_l] = tp / (tp + fp)
        deviation_list.append(d_list)
        gt_num_list.append(g_num)
        dt_num_list.append(d_num)
    ret_dict = {
        'recall': recall,
        'precision': precision,
        'deviation': deviation_list,
        'gt_num': gt_num_list,
        'dt_num': dt_num_list,
        'tp_': tp_num_list,
        'fp_': fp_num_list,
        'fn_': fn_num_list
    }
    # clean temp variables
    del overlaps
    del parted_overlaps
    gc.collect()
    return ret_dict


def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


def bev_box_overlap(boxes, qboxes, criterion=-1):
    riou = iou_rotate_calculate(boxes, qboxes)
    return riou


def iou_rotate_calculate(boxes1, boxes2):
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]
    ious = []
    for i, box1 in enumerate(boxes1):
        temp_ious = []
        r1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
        for j, box2 in enumerate(boxes2):
            r2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])

            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)

                int_area = cv2.contourArea(order_pts)

                inter = int_area * 1.0 / (area1[i] + area2[j] - int_area)
                temp_ious.append(inter)
            else:
                temp_ious.append(0.0)
        ious.append(temp_ious)
    return np.array(ious, dtype=np.float32)


def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50):

    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a['name']) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a['name']) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0
    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        loc = np.concatenate(
            [a['location'][:, :2] for a in gt_annos_part], 0)
        dims = np.concatenate(
            [a['dimensions'][:, [0, 2]] for a in gt_annos_part], 0)
        rots = np.concatenate([a['rotation_y'] for a in gt_annos_part], 0)
        gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                  axis=1)
        loc = np.concatenate(
            [a['location'][:, :2] for a in dt_annos_part], 0)
        dims = np.concatenate(
            [a['dimensions'][:, [0, 2]] for a in dt_annos_part], 0)
        rots = np.concatenate([a['rotation_y'] for a in dt_annos_part], 0)
        dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                  axis=1)
        overlap_part = bev_box_overlap(gt_boxes,
                                       dt_boxes).astype(np.float64)

        parted_overlaps.append(overlap_part)
        example_idx += num_part
    example_idx = 0
    gt_num_idx, dt_num_idx = 0, 0
    overlaps = []
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part
    return overlaps, parted_overlaps, total_gt_num, total_dt_num


def _prepare_data(gt_annos, dt_annos, current_class, dis):
    gt_datas_list = []
    dt_datas_list = []
    gt_datas_list_detail = []
    dt_datas_list_detail = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, dis)
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes, num_valid_dt = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_annos[i]['alpha'] = np.array(gt_annos[i]['alpha'])
        dt_annos[i]['score'] = np.array(dt_annos[i]['score'])
        dt_annos[i]['alpha'] = np.array(dt_annos[i]['alpha'])
        gt_datas = np.concatenate(
            [gt_annos[i]['bbox'], gt_annos[i]['alpha'][..., np.newaxis]], 1)
        dt_datas = np.concatenate([
            dt_annos[i]['bbox'], dt_annos[i]['alpha'][..., np.newaxis],
            dt_annos[i]['score'][..., np.newaxis]
        ], 1)
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
        gt_location = np.array(gt_annos[i]['location'])
        gt_size = np.array(gt_annos[i]['dimensions'])
        gt_ori = np.array(gt_annos[i]['rotation_y'])
        gt_detail = np.concatenate(
            [gt_location, gt_size, gt_ori[..., np.newaxis]], 1)
        dt_location = np.array(dt_annos[i]['location'])
        dt_size = np.array(dt_annos[i]['dimensions'])
        dt_ori = np.array(dt_annos[i]['rotation_y'])
        dt_detail = np.concatenate(
            [dt_location, dt_size, dt_ori[..., np.newaxis]], 1)
        gt_datas_list_detail.append(gt_detail)
        dt_datas_list_detail.append(dt_detail)
    total_dc_num = np.stack(total_dc_num, axis=0)
    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, gt_datas_list_detail, dt_datas_list_detail, dontcares,
            total_dc_num, total_num_valid_gt, num_valid_dt)


def clean_data(gt_anno, dt_anno, current_class, dis):
    CLASS_NAMES = ['Car', 'Van', 'Bus', 'Truck', 'Head', 'Tricycle', 'Vehicle_part',
     'Dynamic_other', 'Pedestrian', 'Bike', 'Cyclist', 'Animal']
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno['name'])
    num_dt = len(dt_anno['name'])
    num_valid_gt = 0
    num_valid_dt = 0
    for i in range(num_gt):
        bbox = gt_anno['bbox'][i]
        gt_name = gt_anno['name'][i].lower()
        valid_class = -1
        if (gt_name == current_cls_name):
            valid_class = 1
        else:
            valid_class = 0
        ignore = False
        if ((gt_anno['location'][i][0] < dis[0]) or (gt_anno['location'][i][0] >= dis[1])):
            ignore = True
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif (valid_class == 0 or (ignore and (valid_class == 1))):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
        if gt_anno['name'][i] == 'DontCare':
            dc_bboxes.append(gt_anno['bbox'][i])
    for i in range(num_dt):
        if (dt_anno['name'][i].lower() == current_cls_name) and (dt_anno['location'][i][0] >= dis[0])\
                and (dt_anno['location'][i][0] < dis[1]):
            valid_class = 1
            ignored_dt.append(0)
            num_valid_dt += 1
        else:
            valid_class = -1
            ignored_dt.append(1)
    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes, num_valid_dt


@numba.jit(nopython=True)
def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           gt_detail,
                           dt_detail,
                           min_overlap,
                           thresh=0,
                           compute_fp=False,
                           compute_aos=False):
    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]
    dt_alphas = dt_datas[:, 4]
    gt_alphas = gt_datas[:, 4]
    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    thresholds = np.zeros((gt_size,))
    thresh_idx = 0
    delta = np.zeros((gt_size,))
    delta_idx = 0
    list_deviation = []
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap)
                    and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            de = {}
            de['center_x'] = dt_detail[det_idx][0] - gt_detail[i][0]
            de['center_y'] = dt_detail[det_idx][1] - gt_detail[i][1]
            de['center_z'] = dt_detail[det_idx][2] - gt_detail[i][2]
            de['Length'] = dt_detail[det_idx][3] - gt_detail[i][3]
            de['Width'] = dt_detail[det_idx][4] - gt_detail[i][4]
            de['Height'] = dt_detail[det_idx][5] - gt_detail[i][5]
            orim = dt_detail[det_idx][6] - gt_detail[i][6]
            if  orim > (3.141593/2.0):
                  orim = orim - 3.141593
            if orim < -(3.141593/2.0):
                  orim = orim + 3.141593
            de['Ori'] = abs(orim)
            list_deviation.append(de)
            if compute_aos:
                delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                delta_idx += 1
            assigned_detection[det_idx] = True
    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0

        fp -= nstuff
        if compute_aos:
            tmp = np.zeros((fp + delta_idx,))
            # tmp = [0] * fp
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx], list_deviation

def rect_loc(row, col, angle, height, bottom):
    xo = np.cos(angle)
    yo = np.sin(angle)
    y1 = row + height / 2 * yo
    x1 = col - height / 2 * xo
    y2 = row - height / 2 * yo
    x2 = col + height / 2 * xo

    return np.array(
        [
         [y1 - bottom/2 * xo, x1 - bottom/2 * yo],
         [y2 - bottom/2 * xo, x2 - bottom/2 * yo],
         [y2 + bottom/2 * xo, x2 + bottom/2 * yo],
         [y1 + bottom/2 * xo, x1 + bottom/2 * yo],
         ]
    ).astype(np.float)


def Process_One_Frame(gt_an, dt_an, current_classes, Is_class_D, name_frame, distance):
    print(name_frame)
    list_pre_de = []
    gt_an_k = {}
    dt_an_k = {}
    gt_an_k = gt_an['label']['3D']
    dt_an_k = dt_an#['data']
    value_re = {}
    value_re['position_x_mean'] = np.zeros(
        [len(current_classes), len(distance)])
    value_re['position_y_mean'] = np.zeros(
        [len(current_classes), len(distance)])
    value_re['position_z_mean'] = np.zeros(
        [len(current_classes), len(distance)])
    value_re['size_l_mean'] = np.zeros(
        [len(current_classes), len(distance)])
    value_re['size_w_mean'] = np.zeros(
        [len(current_classes), len(distance)])
    value_re['size_h_mean'] = np.zeros(
        [len(current_classes), len(distance)])
    value_re['ori'] = np.zeros(
        [len(current_classes), len(distance)])
    value_re['gt_num'] = np.zeros(
        [len(current_classes), len(distance)])
    value_re['dt_num'] = np.zeros(
        [len(current_classes), len(distance)])
    value_re['tp'] = np.zeros(
        [len(current_classes), len(distance)])
    value_re['fp'] = np.zeros(
        [len(current_classes), len(distance)])
    value_re['fn'] = np.zeros(
        [len(current_classes), len(distance)])
    value_re['Is_Valid'] = np.zeros(
        [len(current_classes), len(distance)])
    if len(dt_an_k) == 0:
        return 0, value_re, list_pre_de
    gt_an_p = {}
    gt_an_p['name'] = []
    gt_an_p['location'] = []
    gt_an_p['dimensions'] = []
    gt_an_p['rotation_y'] = []
    gt_an_p['occluded'] = []
    gt_an_p['bbox'] = []
    gt_an_p['alpha'] = []
    gt_an_p['score'] = []
    gt_an_p['truncated'] = []

    dt_an_p = {}
    dt_an_p['name'] = []
    dt_an_p['location'] = []
    dt_an_p['dimensions'] = []
    dt_an_p['rotation_y'] = []
    dt_an_p['occluded'] = []
    dt_an_p['bbox'] = []
    dt_an_p['alpha'] = []
    dt_an_p['score'] = []
    dt_an_p['truncated'] = []

    class_128_to_luminar = {
        'Sedan_Car': 'Car',
        'SUV': 'Car',
        'Bus': 'Bus',
        'BigTruck': 'Truck',
        'Lorry': 'Truck',
        'MiniVan': 'Van',
        'SpecialVehicle': 'Truck',
        'TinyCar': 'Car',
        'Trailer': 'Truck',
        'trailer': 'Truck',
        'EmergencyVehicle': 'Car',
        'Vehicle_others': 'Vehicle_part',
        'Pedestrian': 'Pedestrian',
        'Cycle': 'Bike',
        'Cyclist': 'Cyclist',
        'head': 'Head',
        'Group': 'Group'}

    id_to_class = ['Car', 'Van', 'Bus', 'Truck', 'Head', 'Tricycle', 'Vehicle_part',
     'Dynamic_other', 'Pedestrian', 'Bike', 'Cyclist', 'Animal']
    for index in range(len(gt_an_k)):
        if Is_class_D == True:
            gt_an_k[index]['type'] = class_128_to_luminar[gt_an_k[index]['type']]
            gt_an_p['name'].append(gt_an_k[index]['type'])
        else:
            gt_an_p['name'].append(current_classes[0])
        gt_an_p['location'].append([float(i) for i in gt_an_k[index]['position'].values()])
        gt_an_p['dimensions'].append([float(i) for i in gt_an_k[index]['size']])
        gt_an_p['rotation_y'].append(float(gt_an_k[index]['rotation']['phi']))
        gt_an_p['occluded'].append(0)
        ret = rect_loc(float(gt_an_k[index]['position']['x']), float(gt_an_k[index]['position']['y']),
                       float(gt_an_k[index]['rotation']['phi']), float(gt_an_k[index]['size'][0]),
                       float(gt_an_k[index]['size'][1]))
        gt_an_p['bbox'].append([ret[0][0], ret[0][1], ret[2][0], ret[2][1]])
        gt_an_p['alpha'].append(float(gt_an_k[index]['rotation']['phi']))
        gt_an_p['score'].append(1)
        gt_an_p['truncated'].append(0)
    gt_an_p['location'] = np.array(gt_an_p['location']).reshape(-1, 3)
    gt_an_p['dimensions'] = np.array(gt_an_p['dimensions']).reshape(-1, 3)
    gt_an_p['bbox'] = np.array(gt_an_p['bbox']).reshape(-1, 4)
    nm_dt = 0
    for index in range(len(dt_an_k)):#range(m):#range(m):#range(6):#
        if Is_class_D == True:
            dt_an_k[index]['cls'] = id_to_class[int(dt_an_k[index]['cls'])]
            dt_an_p['name'].append(dt_an_k[index]['cls'])
            nm_dt += 1
        else:
            dt_an_p['name'].append(current_classes[0])
            nm_dt += 1
        dt_an_p['location'].append([float(dt_an_k[index]['x']), float(dt_an_k[index]['y']), float(dt_an_k[index]['z'])])
        dt_an_p['dimensions'].append([float(dt_an_k[index]['l']), float(dt_an_k[index]['w']), float(dt_an_k[index]['h'])])
        dt_an_p['rotation_y'].append(float(dt_an_k[index]['theta']))
        dt_an_p['occluded'].append(0)
        ret = rect_loc(float(dt_an_k[index]['x']), float(dt_an_k[index]['y']),
                       float(dt_an_k[index]['theta']), float(dt_an_k[index]['l']),
                       float(dt_an_k[index]['w']))
        dt_an_p['bbox'].append([ret[0][0], ret[0][1], ret[2][0], ret[2][1]])
        dt_an_p['alpha'].append(float(dt_an_k[index]['theta']))
        dt_an_p['score'].append(float(dt_an_k[index]['score']))
        dt_an_p['truncated'].append(0)
    dt_an_p['location'] = np.array(dt_an_p['location']).reshape(-1, 3)
    dt_an_p['dimensions'] = np.array(dt_an_p['dimensions']).reshape(-1, 3)
    dt_an_p['bbox'] = np.array(dt_an_p['bbox']).reshape(-1, 4)

    dt_an_ps = []
    gt_an_ps = []
    gt_an_ps.append(gt_an_p)
    dt_an_ps.append(dt_an_p)
    if (nm_dt == 0):
        return 0 , value_re, list_pre_de
    ret = saic_od_eval(gt_an_ps, dt_an_ps, current_classes, distance)
#    print('recall ', ret['recall'], 'precision ', ret['precision'])
    Is_Save =False

    from prettytable import PrettyTable
    from imp import reload
    reload(sys)
    title = 'This is normal case'
    table = PrettyTable(['dt-id', 'class', 'distance', 'precision', 'recall', 'loc_mean_x', 'loc_mean_y', 'loc_mean_z',
                         'size_mean_x', 'size_mean_y', 'size_mean_z', 'ori_mean', 'dt_num', 'gt_num'])
    for i in range(len(current_classes)):
        de = []
        for k in range(len(distance)):
            sum_loc_x = 0
            sum_loc_y = 0
            sum_loc_z = 0
            sum_size_l = 0
            sum_size_w = 0
            sum_size_h = 0
            sum_ori = 0
            pre_de = []
            for j in range(len(ret['deviation'][i][k])):
                pre_de.append(ret['deviation'][i][k][j])
                sum_loc_x = sum_loc_x + ret['deviation'][i][k][j]['center_x']
                sum_loc_y = sum_loc_y + ret['deviation'][i][k][j]['center_y']
                sum_loc_z = sum_loc_z + ret['deviation'][i][k][j]['center_z']
                sum_size_l = sum_size_l + ret['deviation'][i][k][j]['Length']
                sum_size_w = sum_size_w + ret['deviation'][i][k][j]['Width']
                sum_size_h = sum_size_h + ret['deviation'][i][k][j]['Height']
                sum_ori = sum_ori + ret['deviation'][i][k][j]['Ori']
            de.append(pre_de)
            if (len(ret['deviation'][i][k]) == 0):
                loc_mean_x = -1
                loc_mean_y = -1
                loc_mean_z = -1
                size_mean_l = -1
                size_mean_w = -1
                size_mean_h = -1
                ori_mean = -1
                value_re['Is_Valid'][i][k] = False
            else:
                loc_mean_x = sum_loc_x / len(ret['deviation'][i][k])
                loc_mean_y = sum_loc_y / len(ret['deviation'][i][k])
                loc_mean_z = sum_loc_z / len(ret['deviation'][i][k])
                size_mean_l = sum_size_l / len(ret['deviation'][i][k])
                size_mean_w = sum_size_w / len(ret['deviation'][i][k])
                size_mean_h = sum_size_h / len(ret['deviation'][i][k])
                ori_mean = sum_ori / len(ret['deviation'][i][k])
                value_re['position_x_mean'][i][k] = sum_loc_x#loc_mean_x
                value_re['position_y_mean'][i][k] = sum_loc_y#loc_mean_y
                value_re['position_z_mean'][i][k] = sum_loc_z#loc_mean_z
                value_re['size_l_mean'][i][k] = sum_size_l#size_mean_l
                value_re['size_w_mean'][i][k] = sum_size_w#size_mean_w
                value_re['size_h_mean'][i][k] = sum_size_h#size_mean_h
                value_re['ori'][i][k] = sum_ori#ori_mean
                value_re['Is_Valid'][i][k] = True
            value_re['gt_num'][i][k] = ret['gt_num'][i][k]
            value_re['dt_num'][i][k] = ret['dt_num'][i][k]
            value_re['tp'][i][k] = ret['tp_'][i][k]
            value_re['fp'][i][k] = ret['fp_'][i][k]
            value_re['fn'][i][k] = ret['fn_'][i][k]
            ret['precision'][i][k] = ('{:.4f}'.format(ret['precision'][i][k]))
            ret['recall'][i][k] = ('{:.4f}'.format(ret['recall'][i][k]))
            loc_mean_x = ('{:.4f}'.format(loc_mean_x))
            loc_mean_y = ('{:.4f}'.format(loc_mean_y))
            loc_mean_z = ('{:.4f}'.format(loc_mean_z))
            size_mean_l = ('{:.4f}'.format(size_mean_l))
            size_mean_w = ('{:.4f}'.format(size_mean_w))
            size_mean_h = ('{:.4f}'.format(size_mean_h))
            ori_mean = ('{:.4f}'.format(ori_mean))
            table.add_row([name_frame, current_classes[i], distance[k], ret['precision'][i][k], ret['recall'][i][k],
                            loc_mean_x, loc_mean_y, loc_mean_z, size_mean_l, size_mean_w, size_mean_h, ori_mean,
                            ret['dt_num'][i][k], ret['gt_num'][i][k]])
            if(float(loc_mean_x)>0.8 or float(loc_mean_y)>0.8 or float(loc_mean_z)>0.8 or float(size_mean_l)>0.8 or
                    float(size_mean_w)>0.8 or float(size_mean_h)>0.8):
                print('This is hard case')
                title = 'This is hard case'
        list_pre_de.append(de)

    print(table)
    if(title == 'This is hard case'):
        out_name_2 = 'hard.txt'
        f = open(out_name_2, "a")
        f.write(name_frame + '\n')
        f.write(str(table) + '\n')
        f.close()
    return 1, value_re, list_pre_de


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="process data")
    parser.add_argument("--gt_root", help="gt_root")
    parser.add_argument("--dt_root", help="dt_root")
    parser.add_argument("--Is_class_D", type=int, default=1)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    gt_root = args.gt_root
    dt_root = args.dt_root
    Is_Class = args.Is_class_D
    root_dt = dt_root#'/home/igs/402/eval/infer_results_0107_model128_5.8w/'
    root_gt = gt_root#'/home/igs/Desktop/eval_test_data/bin/labels/pull/label/'
    Is_class_D = Is_Class#True
    distance = [[-30, 0], [0, 20], [20, 50], [50, 150]]
    if(Is_class_D == True):
        current_classes = ['Car', 'Van', 'Bus', 'Truck', 'Head', 'Tricycle', 'Vehicle_part',
                    'Dynamic_other', 'Pedestrian', 'Bike', 'Cyclist', 'Animal']#['Car', 'Bus', 'Truck']
    else:
        current_classes = ['Car']
    all_dt = [os.path.join(root_dt, one_dt)
               for one_dt in os.listdir(root_dt)]
    val_sum_x = np.zeros(
        [len(current_classes), len(distance)])
    val_sum_y = np.zeros(
        [len(current_classes), len(distance)])
    val_sum_z = np.zeros(
        [len(current_classes), len(distance)])
    val_sum_l = np.zeros(
        [len(current_classes), len(distance)])
    val_sum_w = np.zeros(
        [len(current_classes), len(distance)])
    val_sum_h = np.zeros(
        [len(current_classes), len(distance)])
    val_sum_o = np.zeros(
        [len(current_classes), len(distance)])
    val_sum_tp = np.zeros(
        [len(current_classes), len(distance)])
    val_sum_fp = np.zeros(
        [len(current_classes), len(distance)])
    val_sum_fn = np.zeros(
        [len(current_classes), len(distance)])
    val_sum_valid = np.zeros(
        [len(current_classes), len(distance)])
    val_sum_gt = np.zeros(
        [len(current_classes), len(distance)])
    val_sum_dt = np.zeros(
        [len(current_classes), len(distance)])
    list_std = []
    ct = 0
    for one_dt in all_dt:#[:2]:
        name_frame = one_dt.split('/')[-1][:-5]+'.txt'
        one_gt = root_gt + name_frame
        with open(one_dt, 'r') as f:
            dt_an = json.load(f)
        with open(one_gt, 'r') as f:
            gt_an = json.load(f)
        id = name_frame.split('.')[0]
        Save =False
        Save, value_reru, list_pre_ = Process_One_Frame(gt_an, dt_an, current_classes, Is_class_D, id, distance)
        if Save == 1:
            list_std.append(list_pre_)
        if Save == 0:
            continue
        print('done')
        for i in range(len(current_classes)):
            for k in range(len(distance)):
                if(value_reru['Is_Valid'][i][k] == True):
                    val_sum_x[i][k] += value_reru['position_x_mean'][i][k]
                    val_sum_y[i][k] += value_reru['position_y_mean'][i][k]
                    val_sum_z[i][k] += value_reru['position_z_mean'][i][k]
                    val_sum_l[i][k] += value_reru['size_l_mean'][i][k]
                    val_sum_w[i][k] += value_reru['size_w_mean'][i][k]
                    val_sum_h[i][k] += value_reru['size_h_mean'][i][k]
                    val_sum_o[i][k] += value_reru['ori'][i][k]
                    ct +=1
                    val_sum_valid[i][k] += 1
                else:
                    val_sum_valid[i][k] += 0
                val_sum_tp[i][k] += value_reru['tp'][i][k]
                val_sum_fp[i][k] += value_reru['fp'][i][k]
                val_sum_fn[i][k] += value_reru['fn'][i][k]
                val_sum_gt[i][k] += value_reru['gt_num'][i][k]
                val_sum_dt[i][k] += value_reru['dt_num'][i][k]
    ss = np.zeros([len(current_classes), len(distance), ct,7])
    aa = []
    count = np.zeros(
        [len(current_classes), len(distance)])
    for std in list_std:
        for i in range(len(current_classes)):
            for k in range(len(distance)):
                for m in range(len(std[i][k])):
                    ss[i, k, int(count[i][k]), 0] = float(std[i][k][m]['center_x'])
                    ss[i, k, int(count[i][k]), 1] = float(std[i][k][m]['center_y'])
                    ss[i, k, int(count[i][k]), 2] = float(std[i][k][m]['center_z'])
                    ss[i, k, int(count[i][k]), 3] = float(std[i][k][m]['Length'])
                    ss[i, k, int(count[i][k]), 4] = float(std[i][k][m]['Width'])
                    ss[i, k, int(count[i][k]), 5] = float(std[i][k][m]['Height'])
                    ss[i, k, int(count[i][k]), 6] = float(std[i][k][m]['Ori'])
                    count[i][k]+=1
    std_all = np.zeros(
        [len(current_classes), len(distance),7])
    for i in range(len(current_classes)):
        for k in range(len(distance)):
            if(count[i][k] > 0):
                print(ss[i, k, 0:int(count[i][k]), 0])
                std_all[i, k, 0] = np.std(ss[i, k, 0:int(count[i][k]), 0])
                std_all[i, k, 1] = np.std(ss[i, k, 0:int(count[i][k]), 1])
                std_all[i, k, 2] = np.std(ss[i, k, 0:int(count[i][k]), 2])
                std_all[i, k, 3] = np.std(ss[i, k, 0:int(count[i][k]), 3])
                std_all[i, k, 4] = np.std(ss[i, k, 0:int(count[i][k]), 4])
                std_all[i, k, 5] = np.std(ss[i, k, 0:int(count[i][k]), 5])
                std_all[i, k, 6] = np.std(ss[i, k, 0:int(count[i][k]), 6])
            else:
                std_all[i, k, 0] = 0
    from prettytable import PrettyTable
    from imp import reload
    reload(sys)
    #list_std
    title = 'NN Eval'
    table = PrettyTable(['class', 'distance', 'precision', 'recall', 'meanx', 'stdx', 'meany', 'stdy', 'meanz', 'stdz',
                         'sizel', 'stdl', 'sizew', 'stdw', 'sizeh', 'stdh', 'meano', 'stdo', 'dt_num', 'gt_num'])
    if Is_class_D == False:
        current_classes = ['default']
    for i in range(len(current_classes)):
        for k in range(len(distance)):
            if val_sum_valid[i][k] == 0:
                val_sum_valid[i][k] = 1
            if (val_sum_tp[i][k] + val_sum_fp[i][k]) == 0:
                val_sum_fp[i][k] = 1
            if (val_sum_tp[i][k]+val_sum_fn[i][k]) == 0:
                val_sum_fn[i][k] =1
            table.add_row([current_classes[i],
                           distance[k],
                           ('{:.4f}'.format(val_sum_tp[i][k]/(val_sum_tp[i][k]+val_sum_fp[i][k]))),
                           ('{:.4f}'.format(val_sum_tp[i][k]/(val_sum_tp[i][k]+val_sum_fn[i][k]))),
                           ('{:.4f}'.format(val_sum_x[i][k]/(val_sum_tp[i][k] if val_sum_tp[i][k] > 0 else 1))),
                           ('{:.4f}'.format(std_all[i][k][0])),
                           ('{:.4f}'.format(val_sum_y[i][k]/(val_sum_tp[i][k] if val_sum_tp[i][k] > 0 else 1))),
                           ('{:.4f}'.format(std_all[i][k][1])),
                           ('{:.4f}'.format(val_sum_z[i][k]/(val_sum_tp[i][k] if val_sum_tp[i][k] > 0 else 1))),
                           ('{:.4f}'.format(std_all[i][k][2])),
                           ('{:.4f}'.format(val_sum_l[i][k]/(val_sum_tp[i][k] if val_sum_tp[i][k] > 0 else 1))),
                           ('{:.4f}'.format(std_all[i][k][3])),
                           ('{:.4f}'.format(val_sum_w[i][k]/(val_sum_tp[i][k] if val_sum_tp[i][k] > 0 else 1))),
                           ('{:.4f}'.format(std_all[i][k][4])),
                           ('{:.4f}'.format(val_sum_h[i][k]/(val_sum_tp[i][k] if val_sum_tp[i][k] > 0 else 1))),
                           ('{:.4f}'.format(std_all[i][k][5])),
                           ('{:.4f}'.format(val_sum_o[i][k]/(val_sum_tp[i][k] if val_sum_tp[i][k] > 0 else 1))),
                           ('{:.4f}'.format(std_all[i][k][6])),
                           val_sum_dt[i][k],
                           val_sum_gt[i][k]])
    print(table)
    out_name = 'eval.txt'
    f = open(out_name, "a")
    f.write(title + '\n')
    f.write(str(table)+'\n')
    f.close()



"""python saic_od_eva.py --dt_root /home/igs/402/eval/infer_results_0107_model128_5.8w/ --gt_root /home/igs/Desktop/eval_test_data/bin/labels/pull/label/ --Is_class_D 1"""

