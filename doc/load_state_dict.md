## load_state_dict(state_dict, strict=True)    
Copies parameters and buffers from state_dict into this module and its descendants. If strict is True, then the keys of state_dict must exactly match the keys returned by this module’s state_dict() function.

    device   = next(model.parameters()).device     
    ckpt     = torch.load(file, map_location=device)["state_dict"]    
print(">>>>>>>||", ckpt.keys())  

>>>>>>>|| odict_keys(['conv_input.0.weight', 'conv_input.0._input_quantizer._amax', 'conv_input.0._weight_quantizer._amax', 'conv_input.1.weight', 'conv_input.1.bias', 'conv_input.1.running_mean', 'conv_input.1.running_var', 'conv_input.1.num_batches_tracked', 'conv1.0.conv1.weight', 'conv1.0.conv1.bias', 'conv1.0.conv1._input_quantizer._amax', 'conv1.0.conv1._weight_quantizer._amax', 'conv1.0.conv2.weight', 'conv1.0.conv2.bias', 'conv1.0.conv2._input_quantizer._amax', 'conv1.0.conv2._weight_quantizer._amax', 'conv1.0.quant_add._input_quantizer._amax', 'conv1.1.conv1.weight', 'conv1.1.conv1.bias', 'conv1.1.conv1._input_quantizer._amax', 'conv1.1.conv1._weight_quantizer._amax', 'conv1.1.conv2.weight', 'conv1.1.conv2.bias', 'conv1.1.conv2._input_quantizer._amax', 'conv1.1.conv2._weight_quantizer._amax', 'conv1.1.quant_add._input_quantizer._amax', 'conv2.0.weight', 'conv2.0.bias', 'conv2.0._input_quantizer._amax', 'conv2.0._weight_quantizer._amax', 'conv2.2.conv1.weight', 'conv2.2.conv1.bias', 'conv2.2.conv1._input_quantizer._amax', 'conv2.2.conv1._weight_quantizer._amax', 'conv2.2.conv2.weight', 'conv2.2.conv2.bias', 'conv2.2.conv2._input_quantizer._amax', 'conv2.2.conv2._weight_quantizer._amax', 'conv2.2.quant_add._input_quantizer._amax', 'conv2.3.conv1.weight', 'conv2.3.conv1.bias', 'conv2.3.conv1._input_quantizer._amax', 'conv2.3.conv1._weight_quantizer._amax', 'conv2.3.conv2.weight', 'conv2.3.conv2.bias', 'conv2.3.conv2._input_quantizer._amax', 'conv2.3.conv2._weight_quantizer._amax', 'conv2.3.quant_add._input_quantizer._amax', 'conv3.0.weight', 'conv3.0.bias', 'conv3.0._input_quantizer._amax', 'conv3.0._weight_quantizer._amax', 'conv3.2.conv1.weight', 'conv3.2.conv1.bias', 'conv3.2.conv1._input_quantizer._amax', 'conv3.2.conv1._weight_quantizer._amax', 'conv3.2.conv2.weight', 'conv3.2.conv2.bias', 'conv3.2.conv2._input_quantizer._amax', 'conv3.2.conv2._weight_quantizer._amax', 'conv3.2.quant_add._input_quantizer._amax', 'conv3.3.conv1.weight', 'conv3.3.conv1.bias', 'conv3.3.conv1._input_quantizer._amax', 'conv3.3.conv1._weight_quantizer._amax', 'conv3.3.conv2.weight', 'conv3.3.conv2.bias', 'conv3.3.conv2._input_quantizer._amax', 'conv3.3.conv2._weight_quantizer._amax', 'conv3.3.quant_add._input_quantizer._amax', 'conv4.0.weight', 'conv4.0.bias', 'conv4.0._input_quantizer._amax', 'conv4.0._weight_quantizer._amax', 'conv4.2.conv1.weight', 'conv4.2.conv1.bias', 'conv4.2.conv1._input_quantizer._amax', 'conv4.2.conv1._weight_quantizer._amax', 'conv4.2.conv2.weight', 'conv4.2.conv2.bias', 'conv4.2.conv2._input_quantizer._amax', 'conv4.2.conv2._weight_quantizer._amax', 'conv4.2.quant_add._input_quantizer._amax', 'conv4.3.conv1.weight', 'conv4.3.conv1.bias', 'conv4.3.conv1._input_quantizer._amax', 'conv4.3.conv1._weight_quantizer._amax', 'conv4.3.conv2.weight', 'conv4.3.conv2.bias', 'conv4.3.conv2._input_quantizer._amax', 'conv4.3.conv2._weight_quantizer._amax', 'conv4.3.quant_add._input_quantizer._amax', 'extra_conv.0.weight', 'extra_conv.0.bias', 'extra_conv.0._input_quantizer._amax', 'extra_conv.0._weight_quantizer._amax'])

print(">>>>>>>|| ", model.state_dict().keys())     
>>>>>>>|| odict_keys(['conv_input.0.weight', 'conv_input.1.weight', 'conv_input.1.bias', 'conv_input.1.running_mean', 'conv_input.1.running_var', 'conv_input.1.num_batches_tracked', 'conv1.0.conv1.weight', 'conv1.0.conv1.bias', 'conv1.0.conv2.weight', 'conv1.0.conv2.bias', 'conv1.1.conv1.weight', 'conv1.1.conv1.bias', 'conv1.1.conv2.weight', 'conv1.1.conv2.bias', 'conv2.0.weight', 'conv2.0.bias', 'conv2.2.conv1.weight', 'conv2.2.conv1.bias', 'conv2.2.conv2.weight', 'conv2.2.conv2.bias', 'conv2.3.conv1.weight', 'conv2.3.conv1.bias', 'conv2.3.conv2.weight', 'conv2.3.conv2.bias', 'conv3.0.weight', 'conv3.0.bias', 'conv3.2.conv1.weight', 'conv3.2.conv1.bias', 'conv3.2.conv2.weight', 'conv3.2.conv2.bias', 'conv3.3.conv1.weight', 'conv3.3.conv1.bias', 'conv3.3.conv2.weight', 'conv3.3.conv2.bias', 'conv4.0.weight', 'conv4.0.bias', 'conv4.2.conv1.weight', 'conv4.2.conv1.bias', 'conv4.2.conv2.weight', 'conv4.2.conv2.bias', 'conv4.3.conv1.weight', 'conv4.3.conv1.bias', 'conv4.3.conv2.weight', 'conv4.3.conv2.bias', 'extra_conv.0.weight', 'extra_conv.0.bias'])

print(">>>>>>>>>", model)      
>>>>>>>>> SpMiddleResNetFHD(
  (conv_input): SparseSequential(
    (0): SparseConvolutionQunat(
      5, 16, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm
      (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
      (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
    , act=Activation.None_)
    (1): BatchNorm1d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
  (conv1): SparseSequential(
    (0): SparseBasicBlock(
      (conv1): SparseConvolutionQunat(
        16, 16, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      , act=Activation.None_)
      (relu): ReLU()
      (conv2): SparseConvolutionQunat(
        16, 16, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      , act=Activation.None_)
      (quant_add): QuantAdd(
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      )
    )
    (1): SparseBasicBlock(
      (conv1): SparseConvolutionQunat(
        16, 16, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      , act=Activation.None_)
      (relu): ReLU()
      (conv2): SparseConvolutionQunat(
        16, 16, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      , act=Activation.None_)
      (quant_add): QuantAdd(
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      )
    )
  )
  (conv2): SparseSequential(
    (0): SparseConvolutionQunat(
      16, 32, kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[1, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm
      (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
      (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
    , act=Activation.None_)
    (1): ReLU(inplace=True)
    (2): SparseBasicBlock(
      (conv1): SparseConvolutionQunat(
        32, 32, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      , act=Activation.None_)
      (relu): ReLU()
      (conv2): SparseConvolutionQunat(
        32, 32, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      , act=Activation.None_)
      (quant_add): QuantAdd(
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      )
    )
    (3): SparseBasicBlock(
      (conv1): SparseConvolutionQunat(
        32, 32, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      , act=Activation.None_)
      (relu): ReLU()
      (conv2): SparseConvolutionQunat(
        32, 32, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      , act=Activation.None_)
      (quant_add): QuantAdd(
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      )
    )
  )
  (conv3): SparseSequential(
    (0): SparseConvolutionQunat(
      32, 64, kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[1, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm
      (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
      (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
    , act=Activation.None_)
    (1): ReLU(inplace=True)
    (2): SparseBasicBlock(
      (conv1): SparseConvolutionQunat(
        64, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      , act=Activation.None_)
      (relu): ReLU()
      (conv2): SparseConvolutionQunat(
        64, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      , act=Activation.None_)
      (quant_add): QuantAdd(
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      )
    )
    (3): SparseBasicBlock(
      (conv1): SparseConvolutionQunat(
        64, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      , act=Activation.None_)
      (relu): ReLU()
      (conv2): SparseConvolutionQunat(
        64, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      , act=Activation.None_)
      (quant_add): QuantAdd(
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      )
    )
  )
  (conv4): SparseSequential(
    (0): SparseConvolutionQunat(
      64, 128, kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[0, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm
      (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
      (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
    , act=Activation.None_)
    (1): ReLU(inplace=True)
    (2): SparseBasicBlock(
      (conv1): SparseConvolutionQunat(
        128, 128, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      , act=Activation.None_)
      (relu): ReLU()
      (conv2): SparseConvolutionQunat(
        128, 128, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      , act=Activation.None_)
      (quant_add): QuantAdd(
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      )
    )
    (3): SparseBasicBlock(
      (conv1): SparseConvolutionQunat(
        128, 128, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      , act=Activation.None_)
      (relu): ReLU()
      (conv2): SparseConvolutionQunat(
        128, 128, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      , act=Activation.None_)
      (quant_add): QuantAdd(
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      )
    )
  )
  (extra_conv): SparseSequential(
    (0): SparseConvolutionQunat(
      128, 128, kernel_size=[3, 1, 1], stride=[2, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.MaskImplicitGemm
      (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
      (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
    , act=Activation.None_)
    (1): ReLU()
  )
)

**key**,  **tensor.shape**    
```
conv_input.0.weight, torch.Size([16, 3, 3, 3, 5])
conv_input.0._input_quantizer._amax, torch.Size([])
conv_input.0._weight_quantizer._amax, torch.Size([16, 1, 1, 1, 1])
conv_input.1.weight, torch.Size([16])
conv_input.1.bias, torch.Size([16])
conv_input.1.running_mean, torch.Size([16])
conv_input.1.running_var, torch.Size([16])
conv_input.1.num_batches_tracked, torch.Size([])
conv1.0.conv1.weight, torch.Size([16, 3, 3, 3, 16])
conv1.0.conv1.bias, torch.Size([16])
conv1.0.conv1._input_quantizer._amax, torch.Size([])
conv1.0.conv1._weight_quantizer._amax, torch.Size([16, 1, 1, 1, 1])
conv1.0.conv2.weight, torch.Size([16, 3, 3, 3, 16])
conv1.0.conv2.bias, torch.Size([16])
conv1.0.conv2._input_quantizer._amax, torch.Size([])
conv1.0.conv2._weight_quantizer._amax, torch.Size([16, 1, 1, 1, 1])
conv1.0.quant_add._input_quantizer._amax, torch.Size([])
conv1.1.conv1.weight, torch.Size([16, 3, 3, 3, 16])
conv1.1.conv1.bias, torch.Size([16])
conv1.1.conv1._input_quantizer._amax, torch.Size([])
conv1.1.conv1._weight_quantizer._amax, torch.Size([16, 1, 1, 1, 1])
conv1.1.conv2.weight, torch.Size([16, 3, 3, 3, 16])
conv1.1.conv2.bias, torch.Size([16])
conv1.1.conv2._input_quantizer._amax, torch.Size([])
conv1.1.conv2._weight_quantizer._amax, torch.Size([16, 1, 1, 1, 1])
conv1.1.quant_add._input_quantizer._amax, torch.Size([])
conv2.0.weight, torch.Size([32, 3, 3, 3, 16])
conv2.0.bias, torch.Size([32])
conv2.0._input_quantizer._amax, torch.Size([])
conv2.0._weight_quantizer._amax, torch.Size([32, 1, 1, 1, 1])
conv2.2.conv1.weight, torch.Size([32, 3, 3, 3, 32])
conv2.2.conv1.bias, torch.Size([32])
conv2.2.conv1._input_quantizer._amax, torch.Size([])
conv2.2.conv1._weight_quantizer._amax, torch.Size([32, 1, 1, 1, 1])
conv2.2.conv2.weight, torch.Size([32, 3, 3, 3, 32])
conv2.2.conv2.bias, torch.Size([32])
conv2.2.conv2._input_quantizer._amax, torch.Size([])
conv2.2.conv2._weight_quantizer._amax, torch.Size([32, 1, 1, 1, 1])
conv2.2.quant_add._input_quantizer._amax, torch.Size([])
conv2.3.conv1.weight, torch.Size([32, 3, 3, 3, 32])
conv2.3.conv1.bias, torch.Size([32])
conv2.3.conv1._input_quantizer._amax, torch.Size([])
conv2.3.conv1._weight_quantizer._amax, torch.Size([32, 1, 1, 1, 1])
conv2.3.conv2.weight, torch.Size([32, 3, 3, 3, 32])
conv2.3.conv2.bias, torch.Size([32])
conv2.3.conv2._input_quantizer._amax, torch.Size([])
conv2.3.conv2._weight_quantizer._amax, torch.Size([32, 1, 1, 1, 1])
conv2.3.quant_add._input_quantizer._amax, torch.Size([])
conv3.0.weight, torch.Size([64, 3, 3, 3, 32])
conv3.0.bias, torch.Size([64])
conv3.0._input_quantizer._amax, torch.Size([])
conv3.0._weight_quantizer._amax, torch.Size([64, 1, 1, 1, 1])
conv3.2.conv1.weight, torch.Size([64, 3, 3, 3, 64])
conv3.2.conv1.bias, torch.Size([64])
conv3.2.conv1._input_quantizer._amax, torch.Size([])
conv3.2.conv1._weight_quantizer._amax, torch.Size([64, 1, 1, 1, 1])
conv3.2.conv2.weight, torch.Size([64, 3, 3, 3, 64])
conv3.2.conv2.bias, torch.Size([64])
conv3.2.conv2._input_quantizer._amax, torch.Size([])
conv3.2.conv2._weight_quantizer._amax, torch.Size([64, 1, 1, 1, 1])
conv3.2.quant_add._input_quantizer._amax, torch.Size([])
conv3.3.conv1.weight, torch.Size([64, 3, 3, 3, 64])
conv3.3.conv1.bias, torch.Size([64])
conv3.3.conv1._input_quantizer._amax, torch.Size([])
conv3.3.conv1._weight_quantizer._amax, torch.Size([64, 1, 1, 1, 1])
conv3.3.conv2.weight, torch.Size([64, 3, 3, 3, 64])
conv3.3.conv2.bias, torch.Size([64])
conv3.3.conv2._input_quantizer._amax, torch.Size([])
conv3.3.conv2._weight_quantizer._amax, torch.Size([64, 1, 1, 1, 1])
conv3.3.quant_add._input_quantizer._amax, torch.Size([])
conv4.0.weight, torch.Size([128, 3, 3, 3, 64])
conv4.0.bias, torch.Size([128])
conv4.0._input_quantizer._amax, torch.Size([])
conv4.0._weight_quantizer._amax, torch.Size([128, 1, 1, 1, 1])
conv4.2.conv1.weight, torch.Size([128, 3, 3, 3, 128])
conv4.2.conv1.bias, torch.Size([128])
conv4.2.conv1._input_quantizer._amax, torch.Size([])
conv4.2.conv1._weight_quantizer._amax, torch.Size([128, 1, 1, 1, 1])
conv4.2.conv2.weight, torch.Size([128, 3, 3, 3, 128])
conv4.2.conv2.bias, torch.Size([128])
conv4.2.conv2._input_quantizer._amax, torch.Size([])
conv4.2.conv2._weight_quantizer._amax, torch.Size([128, 1, 1, 1, 1])
conv4.2.quant_add._input_quantizer._amax, torch.Size([])
conv4.3.conv1.weight, torch.Size([128, 3, 3, 3, 128])
conv4.3.conv1.bias, torch.Size([128])
conv4.3.conv1._input_quantizer._amax, torch.Size([])
conv4.3.conv1._weight_quantizer._amax, torch.Size([128, 1, 1, 1, 1])
conv4.3.conv2.weight, torch.Size([128, 3, 3, 3, 128])
conv4.3.conv2.bias, torch.Size([128])
conv4.3.conv2._input_quantizer._amax, torch.Size([])
conv4.3.conv2._weight_quantizer._amax, torch.Size([128, 1, 1, 1, 1])
conv4.3.quant_add._input_quantizer._amax, torch.Size([])
extra_conv.0.weight, torch.Size([128, 3, 1, 1, 128])
extra_conv.0.bias, torch.Size([128])
extra_conv.0._input_quantizer._amax, torch.Size([])
extra_conv.0._weight_quantizer._amax, torch.Size([128, 1, 1, 1, 1])

```
