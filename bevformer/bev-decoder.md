```
2024-05-04 21:29:31.305 | INFO     | projects.mmdet3d_plugin.bevformer.modules.decoder:forward:101 - lid 0 layer DetrTransformerDecoderLayer(
  (attentions): ModuleList(
    (0): MultiheadAttention(
      (attn): MultiheadAttention(
        (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
      )
      (proj_drop): Dropout(p=0.0, inplace=False)
      (dropout_layer): Dropout(p=0.1, inplace=False)
    )
    (1): CustomMSDeformableAttention(
      (dropout): Dropout(p=0.1, inplace=False)
      (sampling_offsets): Linear(in_features=256, out_features=64, bias=True)
      (attention_weights): Linear(in_features=256, out_features=32, bias=True)
      (value_proj): Linear(in_features=256, out_features=256, bias=True)
      (output_proj): Linear(in_features=256, out_features=256, bias=True)
    )
  )
  (ffns): ModuleList(
    (0): FFN(
      (activate): ReLU(inplace=True)
      (layers): Sequential(
        (0): Sequential(
          (0): Linear(in_features=256, out_features=512, bias=True)
          (1): ReLU(inplace=True)
          (2): Dropout(p=0.1, inplace=False)
        )
        (1): Linear(in_features=512, out_features=256, bias=True)
        (2): Dropout(p=0.1, inplace=False)
      )
      (dropout_layer): Identity()
    )
  )
  (norms): ModuleList(
    (0): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
    (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
    (2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
  )
)
```
