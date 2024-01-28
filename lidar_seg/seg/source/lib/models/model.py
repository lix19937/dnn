from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from .networks.large_hourglass import get_large_hourglass_net
from .networks.resnet import get_resnet34
from .networks.unet import get_unet
from .networks.salsa_next_v4 import get_salsa

from trains.lr_scheduler import update_lr
from loguru import logger

_model_factory = {
    'hourglass': get_large_hourglass_net,
    'resnet34': get_resnet34,
    'unet': get_unet,
    'salsa': get_salsa,
}

def create_model_quan(opt, phase='train', quant_nn=None):
    logger.info("create_model phase:{}, qdq:{}".format(phase, opt.qdq))
    arch = opt.arch
    seg_class_num = opt.ignore_index

    get_model = _model_factory[arch] 
    model = get_model(seg_class=seg_class_num, phase=phase, quantize=True, quant_nn=quant_nn)
    return model

def create_model(opt, phase='train'):
    arch = opt.arch
    seg_class_num = opt.ignore_index

    get_model = _model_factory[arch]
    model = get_model(seg_class=seg_class_num, phase=phase)
    return model


def load_model(model, model_path, optimizer=None, resume=False,
               lr=None, num_epochs=400, user_spec=False):

    logger.info("load_model model_path:{}".format(model_path))           
    if user_spec:
        checkpoint = torch.load(
        model_path, map_location=lambda storage, loc: storage)

        if not isinstance(checkpoint, dict):
            print(checkpoint.__dict__.keys())
            model.load_state_dict(checkpoint.state_dict())
            return model

        if "model" in checkpoint.keys():
          model.load_state_dict(checkpoint["model"])
        else:
          model.load_state_dict(checkpoint['state_dict'],  strict=False )  # strict=False         
        if optimizer is not None:
            return model, optimizer, 0
        else:
            return model

    start_epoch = 0
    checkpoint = torch.load(
        model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx ' + \
          'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, '
                      'loaded shape{}. {}'.format(
                          k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = update_lr(start_epoch, 10, 1e-5,
                                 num_epochs, lr, 0.8)
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')

    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def save_model(path, epoch, model, optimizer=None):
    if 'downCntx.conv1._input_quantizer._amax' in model.state_dict():
        logger.info("save_model, model:{}, epoch:{}\nconv1._input_quantizer\n{}\nconv1._weight_quantizer\n{}\
            \nconv1.bias\n{}\nconv2._input_quantizer\n{}\nconv2._weight_quantizer\n{}\nconv2.bias\n{}".format(path, epoch, 
            model.state_dict()['downCntx.conv1._input_quantizer._amax'].view(1,-1), 
            model.state_dict()['downCntx.conv1._weight_quantizer._amax'].view(1,-1),
            model.state_dict()['downCntx.conv1.bias'].view(1,-1),
            model.state_dict()['downCntx.conv2._input_quantizer._amax'].view(1,-1),
            model.state_dict()['downCntx.conv2._weight_quantizer._amax'].view(1,-1),
            model.state_dict()['downCntx.conv2.bias'].view(1,-1)))
    else:
        logger.info("save_model, model:{}, epoch:{}".format(path, epoch))

    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)
