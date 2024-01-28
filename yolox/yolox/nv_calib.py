

import cv2
from datetime import datetime as dt
from glob import glob
import numpy as np
import os

import sys
import torch 
from torch.autograd import Variable
from loguru import logger

sys.path.append('..')

#-----------------------------------------------------------------
from yolox.nv_qdq import QDQ
#-----------------------------------------------------------------

from yolox.exp import get_exp_by_absfile
from yolox.models.components.network_blocks import SiLU
from yolox.utils import replace_module
from tqdm import tqdm

np.random.seed(97)
torch.manual_seed(97)
torch.cuda.manual_seed_all(97)
torch.backends.cudnn.deterministic = True

nTrainBatchSize = 64

#dataPath = "/home/igs/fusion_se_sod_training-QAT/datasets/"
dataPath = '/Data/tet/fusion_se_sod_finetune/datasets/'

trainFileList = sorted(glob(dataPath + "train/image_height/*.png"))
testFileList  = sorted(glob(dataPath + "val/image_height/*.png"))

calibrator = ["max", "histogram"][1]
percentileList = [99.9, 99.99, 99.999, 99.9999]

np.set_printoptions(precision=4, linewidth=200, suppress=True)

def preproc(img, input_size=[640, 640], swap=(2, 0, 1), mean=[127, 107, 153], std=[56.5115, 29.692, 49.4464]):
    padded_img = np.zeros((input_size[0], input_size[1], 3), dtype=np.float32)

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    if mean is not None and std is not None:
        padded_img -= np.array(mean).reshape(1, 1, 3)
        padded_img /= np.array(std).reshape(1, 1, 3)

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img

class MyData(torch.utils.data.Dataset):
    def __init__(self, isTrain=True):
        if isTrain:
            self.data = trainFileList
        else:
            self.data = testFileList

    def __getitem__(self, index):
        imageName = self.data[index]
        data = cv2.imread(imageName)
        data = preproc(data)
        label = np.zeros(10, dtype=np.float32)
        return torch.from_numpy(data), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)

trainDataset = MyData(True)
testDataset = MyData(False)
trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=nTrainBatchSize, num_workers=4, pin_memory=True, shuffle=True)
# testLoader = torch.utils.data.DataLoader(dataset=testDataset, batch_size=nTrainBatchSize, num_workers=4, pin_memory=True, shuffle=True)

def make_model(ckpt, exp_file):
    exp = get_exp_by_absfile(exp_file)

    model = exp.get_model(head_type='obb').cuda()
    ckpt_file = ckpt

    ckpt = torch.load(ckpt_file, map_location="cuda")
    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt, strict=False)
    model.head.decode_in_inference = False
    logger.info("make_model done")
    return model

def ptq_calib(model, ptq_out_path, trainLoader, nCalibrationBatch):
  logger.info("calib ...")
  with torch.no_grad():
      logger.info("open calib ...")
      for _, module in model.named_modules():
          if isinstance(module, QDQ.quant_nn.TensorQuantizer):
              if module._calibrator is not None:
                  module.disable_quant()
                  module.enable_calib()
              else:
                  module.disable()
      
      logger.info("calib infer ...")
      for i, (xTrain, _) in tqdm(enumerate(trainLoader), total=nCalibrationBatch):
          if i >= nCalibrationBatch:
              break
          model(Variable(xTrain).cuda())

      logger.info("close calib ...")
      for _, module in model.named_modules():
          if isinstance(module, QDQ.quant_nn.TensorQuantizer):
              if module._calibrator is not None:
                  module.enable_quant()
                  module.disable_calib()
              else:
                  module.enable()

      def computeArgMax(model, **kwargs):
          for name, module in model.named_modules():
              if isinstance(module, QDQ.quant_nn.TensorQuantizer) and module._calibrator is not None:
                  if isinstance(module._calibrator, QDQ.calib.MaxCalibrator):
                      module.load_calib_amax()
                  else:
                      module.load_calib_amax(**kwargs)
                  print(F"{name:40}: {module}")

      logger.info("computeArgMax ...")
      if calibrator == "max":
          computeArgMax(model, method="max")
          modelName = "model-max-%d.pth" % (nCalibrationBatch * trainLoader.batch_size)
      else:
          logger.info("computeArgMax percentile ...")
          for percentile in percentileList:
              computeArgMax(model, method="percentile")
              modelName = "model-percentile-%f-%d.pth" % (percentile, nCalibrationBatch * trainLoader.batch_size)
          
          logger.info("computeArgMax mse entropy ...")
          for method in ["mse", "entropy"]:
              computeArgMax(model, method=method)
              modelName = "model-%s-%f.pth" % (method, percentile)

      if not os.path.exists(ptq_out_path):    
          os.makedirs(ptq_out_path)     

      torch.save(model.state_dict(), os.path.join(ptq_out_path, modelName))
  logger.info("Succeeded calibrating model in pyTorch!")


if __name__ == "__main__":
  #ckpt_file ='/home/igs/fusion_se_sod_training-QAT/YOLOX_outputs/yolox_obb_s_fp32/epoch_299_ckpt.pth'
  #exp_file = '/home/igs/fusion_se_sod_training-QAT/exps/example/yolox_obb/yolox_obb_s.py'

  ckpt_file = '/Data/tet/fusion_se_sod_finetune/YOLOX_outputs/yolox_obb_s_fp32/epoch_299_ckpt.pth'
  exp_file  = '/Data/tet/fusion_se_sod_finetune/exps/example/yolox_obb/yolox_obb_s.py'
  model = make_model(ckpt=ckpt_file, exp_file=exp_file)

  ptq_calib(model, ptq_out_path='ptq_out', trainLoader=trainLoader, nCalibrationBatch=2) 
