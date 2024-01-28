from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import torch
from datasets.dataset_factory import get_dataset
from models.model import create_model, load_model
from opts import opts
import numpy as np


def main(opt, pth_path, verify=True):
    # torch.backends.cudnn.benchmark = True
    model = create_model(opt, 'val')
    model = load_model(model, pth_path)
    model = model.to('cuda')
    model.eval()
    Dataset = get_dataset('luminar', 'rv_seg')
    data = Dataset(opt, 'val')[0]
    dummy_input = torch.tensor(data['input'][None, ...], device='cuda')
    input_names = ['input']
    output_names = ['output_seg', 'output_prob']
    # Uses opset version 11 to support masked_select.
    torch.onnx.export(model, (dummy_input), 'lidarnet_seg.onnx', verbose=True,
                      input_names=input_names, output_names=output_names, opset_version=13,
                      do_constant_folding=True)
                    
    # check by onnx
    if verify:
        import onnx
        import onnxruntime as rt
        onnx_model = onnx.load('lidarnet_seg.onnx')
        onnx.checker.check_model(onnx_model)
        # check the numerical value
        # get pytorch output
        with torch.no_grad():
            pytorch_result = model(dummy_input)[0]

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 1)
        sess = rt.InferenceSession('lidarnet_seg.onnx')
        mm_inputs = {"input": dummy_input}
        # mm_inputs = {"input": dummy_input}
        onnx_result = sess.run(
            None, {net_feed_input[0]: mm_inputs[net_feed_input[0]].cpu().numpy()})
        pytorch_result = [pytorch_result['seg'].cpu().numpy()]
        # compare results
        np.testing.assert_allclose(
            pytorch_result[0].astype(np.float32),
            onnx_result[0].astype(np.float32),
            rtol=1e-5,
            atol=1e-5,
            err_msg='The outputs are different between Pytorch and ONNX')
        print('The outputs are same between Pytorch and ONNX')


if __name__ == '__main__':
    opt = opts().parse()
    assert opt.load_model != '', \
    "load_model can't be empty"
    main(opt, opt.load_model)

"""python export_onnx.py lidar_od_seg --num_stacks 1or 2 --dataset segmentation --load_model {model_path}"""
