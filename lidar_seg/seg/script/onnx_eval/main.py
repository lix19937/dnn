import onnx
import onnxruntime as rt
import os
import sys
import os.path as osp
import numpy as np
from tqdm import tqdm
from seg_data import RvSeg
import queue
import seg_eval
import time

import signal
from trt_inference import TRT

# 自定义信号处理函数
def my_handler(signum, frame):
    global stop
    stop = True

# 设置相应信号处理的handler
signal.signal(signal.SIGINT, my_handler)    #读取Ctrl+c信号

gt_map = {0:0, 1:1, 2:2, 3:2, 4:2, 5:2, 6:3, 7:4, 8:2, 9:5, 10:6,
          11:2, 12:7, 13:8, 14:2, 15:3, 16:3, 17:9, 18:9, 19:9, 20:9, 21:9}

def infer_onnx(sess, dummy_input):
    mm_inputs = {"input": dummy_input}
    onnx_result = sess.run(None, mm_inputs)
    return onnx_result

def eval_infer(data_dir, onnx_path):
    bin_dir = osp.join(data_dir, 'pcd_dir')
    bin_list = sorted(os.listdir(bin_dir))
    global stop
    stop = False
    
    ignore = 9
    total_hist = [np.zeros((ignore, ignore), dtype=np.int64) for _ in range(4)]
    worst_status = queue.PriorityQueue()

    segData = RvSeg(17)
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    #sess = rt.InferenceSession(onnx_path)
    mytrt = TRT(onnx_path)
    result = {}
    for bin_file in tqdm(bin_list):
        if stop:
            break
        bin_path = osp.join(bin_dir, bin_file)
        # pc = np.fromfile(bin_path, np.float32).reshape(-1, 3)
        img, gt = segData.project_to_img(bin_path)
        result['raw_points'] = img[0]
        #infer_result = infer_onnx(sess, img)####
        infer_result = mytrt.inference_heads(img)
        gt = [gt_map[_] for _ in gt.flatten()]
        result['gt_label'] = np.array(gt)
        # print(type(infer_result), len(infer_result))
        # print(type(infer_result[0]), infer_result[0].shape, infer_result[0].dtype)

        result['predict_labels'] = infer_result[0]
        result['path'] = bin_path
        inds_1 = (img[0][0, :] <= 30) & (img[0][3, :] > 0.1)
        inds_2 = (img[0][0, :] > 30) & (img[0][0, :] <= 50) & (img[0][3, :] > 0.1)
        inds_3 = (img[0][0, :] > 50) & (img[0][3, :] > 0.1)
        seg_eval.calc_hist(result, (inds_1, inds_2, inds_3), ignore, total_hist, worst_status)
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    root_dir = os.path.join(data_dir, f'eval_{time_str}')
    seg_eval.evaluate(worst_status, total_hist, root_dir, ignore, None)


if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     print(f"usage: python {__file__} DATA_DIR")
    #     exit(-1)
    # root_dir = sys.argv[1]
    root_dir = "/home/igs/seg_train_nfs/lidar_data/single_seg_0901"

    onnx_file = '/home/igs/seg_train_nfs/seg/source/quantization/nqdq_1129.onnx'
    onnx_qat = '/home/igs/seg_train_nfs/seg/source/ptq_calib_out_v2/model-entropy-99.999900.onnx'

    onnx_qat = '/home/igs/seg_train_nfs/seg/model_1129.onnx'

    eval_infer(root_dir, onnx_qat)
