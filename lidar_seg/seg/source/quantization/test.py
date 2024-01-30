# /**************************************************************
#  * @Copyright: 2021-2022 Copyright 
#  * @Author: lix
#  * @Date: 2022-09-03 11:09:48
#  * @Last Modified by: lix
#  * @Last Modified time: 2022-09-03 11:09:48
#  **************************************************************/

import onnx
from loguru import logger
from rules import *

import sys
sys.path.append('./')
from quantize_lx import *
from salsa_next_v4_with_shoartcut import SalsaNext
from glob import glob
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import hiddenlayer as h
from torchsummary import summary

# import imp
# imp.find_module('pytorch_quantization')

# https://blog.csdn.net/dou3516/article/details/125975967
# https://blog.csdn.net/weixin_42233605/article/details/125214041

rand_seed = 123456
np.random.seed(rand_seed)
torch.manual_seed(rand_seed)
torch.cuda.manual_seed_all(rand_seed)

def test_1():
    bpp = '../'
    onnx_file = "lidarnet_seg_cba.onnx"

    onnx_model = onnx.load(bpp + onnx_file)
    graph = onnx_model.graph
    node = graph.node
    for it in node:
      if it.op_type == "Concat":
        logger.info("{}".format(it.output[0]))
        break

def test_2():
    module_dict = {}
    for entry in quant_modules._DEFAULT_QUANT_MAP:
        #logger.info("entry:{}".format(entry))
        module = getattr(entry.orig_mod, entry.mod_name)
        # logger.info("entry:{}".format(entry.orig_mod)) # module 'torch.nn'  
        logger.info("entry:{}".format(entry.mod_name)) # Conv2d
        logger.info("entry:{}".format(module)) # torch.nn.modules.conv.Conv2d
        logger.info("entry:{}".format(entry.replace_mod)) # pytorch_quantization.nn.modules.quant_conv.QuantConv2d

        module_dict[id(module)] = entry.replace_mod

    logger.info("entry:{}".format(module_dict.values()))

def export_onnx_nqdq(model, input, file, *args, **kwargs):
    model.eval()
    with torch.no_grad():
        tmp = "tmp.onnx"
        torch.onnx.export(model, input, tmp, *args, **kwargs)
        logger.info("export onnx done !")
        from onnxsim import simplify 
        import onnx, os
        model = onnx.load(tmp)
        os.remove(tmp)

        onnx.checker.check_model(model)       
        model_simp, check = simplify(model) 
        assert check, "Simplified ONNX model could not be validated"
        onnx.checker.check_model(model_simp)
        onnx.save(model_simp, file)
        logger.info("simplify onnx done !")

def myexport_onnx_nqdq(model, file, size, dynamic_batch=False):
      device = next(model.parameters()).device
      model.float()
      dummy = torch.zeros(1, size[0], size[1], size[2], device=device)

      export_onnx_nqdq(model, dummy, file, opset_version=13, # op=11 vs 13  softmax behavior is diff
          input_names=["images"], 
          # output_names=["output"], 
          # dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}} if dynamic_batch else None
      )   

def export_onnx_qdq(model, file, size, dynamic_batch=False):
    device = next(model.parameters()).device
    model.float()
    model.cuda()
    dummy = torch.zeros(1, size[0], size[1], size[2], device=device).cuda()

    export_onnx(model, dummy, file, opset_version=13, 
        input_names=["images"], 
        # output_names=["outputs1", "outputs2"], 
        # dynamic_axes={"images": {0: "batch"}, "outputs1": {0: "batch"}, "outputs2": {0: "batch"}} if dynamic_batch else None
    )


def test_3(ckpt = None) -> None:
    # Super Resolution model definition in PyTorch

    class SuperResolutionNet(nn.Module):
        def __init__(self, upscale_factor:int, inplace:bool=False, quantize:bool = False, quant_nn = None):
            super(SuperResolutionNet, self).__init__()
    
            self.relu = nn.ReLU(inplace=inplace)
            if quantize:
              self.conv1 = quant_nn.QuantConv2d(1, 64, (5, 5), (1, 1), (2, 2))
              self.conv2 = quant_nn.QuantConv2d(64, 64, (3, 3), (1, 1), (1, 1))
              self.conv3 = quant_nn.QuantConv2d(128, 32, (3, 3), (1, 1), (1, 1))
              self.conv4 = quant_nn.QuantConv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
              self.avgpool1 = quant_nn.QuantAvgPool2d(kernel_size=(3, 3), stride=1, padding=0, count_include_pad=True)# 2, 1; 1, 0
            else:
              self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
              self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
              self.conv3 = nn.Conv2d(128, 32, (3, 3), (1, 1), (1, 1))
              self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
              self.avgpool1 = nn.AvgPool2d(kernel_size=(3, 3), stride=1, padding=0, count_include_pad=True)# 2, 1; 1, 0

            self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

            self._initialize_weights()
            # Layers of non learning parameters, but can quantize. User must set the layer name according to the forward data flow !!! 
            self.LONLP = ['avgpool1']

        def forward(self, x):
            x1 = self.relu(self.conv1(x))
            x2 = self.relu(self.conv2(x1))

            concat = torch.cat((x1, x2), dim=1)

            x = self.avgpool1(concat)
            #x = concat
            x = self.relu(self.conv3(x))
            x = self.pixel_shuffle(self.conv4(x))
            return x
    
        def _initialize_weights(self):
            init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
            init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
            init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
            init.orthogonal_(self.conv4.weight)
            init.zeros_(self.conv4.bias)
    
    _, quant_nn = initialize()
    exec_manual_quantize = True
    exec_rules_quantize = not exec_manual_quantize 
    exec_lidar_model = True
    exec_summary = False

    save_prefix = ''
    if not exec_lidar_model:
      save_prefix = 'SuperResolutionNet'
      model = SuperResolutionNet(upscale_factor=3, quantize = exec_manual_quantize, quant_nn = quant_nn); sz = (1, 224, 224)
      logger.info(model.__dict__['_modules'].keys())
      logger.info(model.state_dict().keys());
    else:  
      save_prefix = 'SalsaNext_shortcut'
      model = SalsaNext(nclasses=9, phase="eval", quantize=exec_manual_quantize, quant_nn = quant_nn); sz = (5, 192, 1024)
      model.LONLP = ['resBlock4.pool', 'resBlock3.pool', 'resBlock2.pool', 'resBlock1.pool']

    if ckpt is not None and ckpt != '':  
        pth_file = ckpt
        try:
          checkpoint = torch.load(pth_file)
        except:
          logger.error("torch.load error {}".format(pth_file))
          exit(0)

        # logger.info("{}".format(checkpoint.keys())) 

        model.load_state_dict(checkpoint)
        # logger.info(model.state_dict)
        
        export_onnx_file = pth_file.replace(".pth", "_reload.onnx")
        export_onnx_qdq(model, export_onnx_file, sz, dynamic_batch=False)
        exit(0)

    device_raw = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_raw)
    model = model.to(device)
    if exec_summary:
        summary(model, input_size=sz, device=device_raw)
    
    # vis_graph = h.build_graph(model, torch.zeros([1, 5, 192, 1024]).cuda())    
    # vis_graph.theme = h.graph.THEMES["blue"].copy()      
    # vis_graph.save("./graph.png")     

    if not exec_manual_quantize and not exec_rules_quantize:
        myexport_onnx_nqdq(model, save_prefix + "_nqdq.onnx", size=(5, 192, 1024), dynamic_batch=False);return
    
    if exec_rules_quantize:
        replace_to_quantization_module(model, ignore_policy="conv996") ### s1

    def myexport_onnx(model, file, size=sz, dynamic_batch=False):
      device = next(model.parameters()).device
      model.float()
      dummy = torch.zeros(1, size[0], size[1], size[2], device=device)
      export_onnx(model, dummy, file, opset_version=13, 
          input_names=["images"], 
          output_names=["output"], 
          dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}} if dynamic_batch else None
      )

    LONLP = model.LONLP if hasattr(model, 'LONLP') else []           
    apply_custom_rules_to_quantizer(model, myexport_onnx, (5, 192, 1024), lonlp = LONLP) ### s2 
    logger.info(model.state_dict().keys()) 

    dataPath = './datasets/'
    trainFileList = sorted(glob(dataPath + "train/*.jpg"))
    testFileList  = sorted(glob(dataPath + "val/*.jpg"))

    class MyData(torch.utils.data.Dataset):
      def __init__(self, isTrain=True):
        if isTrain:
            self.data = trainFileList
        else:
            self.data = testFileList

      def __getitem__(self, index):
          data = np.random.rand(sz[0], sz[1], sz[2])
          label = np.zeros(10, dtype=np.float32)
          return torch.from_numpy(data), torch.from_numpy(label)

      def __len__(self):
          return len(self.data)

    trainDataset = MyData(True)
    train_dataloader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=1, num_workers=4, pin_memory=True, shuffle=True)

    calibrate_model(model, train_dataloader, device, num_batch=4) ###

    calib_ckpt_path = "ptq/"+ save_prefix + "calib.pth"
    export_onnx_file = calib_ckpt_path.replace(".pth", ".onnx")
    torch.save(model.state_dict(), calib_ckpt_path)

    export_onnx_qdq(model, export_onnx_file, size=(sz[0], sz[1], sz[2]))
    logger.info("export_onnx done {}".format(export_onnx_file))


if __name__ == "__main__":
    logger.info('onnx version:{}'.format(onnx.__version__))
    logger.info(torch.__version__) ##  RTX3070 1.8.0+cu111    torch-1.10.2

    test_3()
    save_prefix = 'SalsaNext_shortcut'
    ckpt = "./ptq/"+ save_prefix + "calib.pth"   
    test_3(ckpt)
    logger.info("done")
   