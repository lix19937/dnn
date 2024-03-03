## gpt2    
|overview &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;| decoder&emsp;|      
![gpt2](https://github.com/lix19937/history/assets/38753233/8c8758fc-ca4d-4225-84b3-5e6b8551759f)    

### gpt-block   
![gpt-block](https://github.com/lix19937/history/assets/38753233/4240c50b-2bf3-40bb-aec5-51d850092202)
   
###  gpt-attention-no-cache
![gpt-attention-no-cache](https://github.com/lix19937/history/assets/38753233/7ecff23e-6492-4ece-ae25-c6444a86a613)

> #### Conv1D     Basically works like a linear layer but the weights are transposed.
```
class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    # bias + x * weight  
    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        # https://pytorch.org/docs/stable/generated/torch.addmm.html
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x
```
### gpt-attention-use-cache   
![gpt-attention-use-cache](https://github.com/lix19937/history/assets/38753233/db529cbc-84f5-49c5-ae12-8e7a51c201bd)

### no use kv cache   
```
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("./gpt2", torchscript=True).eval()

# tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2")
in_text = "Lionel Messi is a"
in_tokens = torch.tensor(tokenizer.encode(in_text))

# inference
token_eos = torch.tensor([198]) # line break symbol
out_token = None
i = 0
with torch.no_grad():
    while out_token != token_eos:
        logits, _ = model(in_tokens)
        out_token = torch.argmax(logits[-1, :], dim=0, keepdim=True)
        in_tokens = torch.cat((in_tokens, out_token), 0)# 将当前结果拼接到当前输入 token 后面，形成新的输入 token
        text = tokenizer.decode(in_tokens)
        print(f'step {i} input: {text}', flush=True)
        i += 1

out_text = tokenizer.decode(in_tokens)
print(f' Input: {in_text}')
print(f'Output: {out_text}')
```

### use kv cache 
```
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("./gpt2", torchscript=True).eval()

# tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2")
in_text = "Lionel Messi is a"
in_tokens = torch.tensor(tokenizer.encode(in_text))

# inference
token_eos = torch.tensor([198]) # line break symbol
out_token = None

kvcache = None
out_text = in_text

i = 0
with torch.no_grad():
    while out_token != token_eos:
        logits, kvcache = model(in_tokens, past_key_values=kvcache) # 增加了一个 past_key_values 的参数
        out_token = torch.argmax(logits[-1, :], dim=0, keepdim=True)
        in_tokens = out_token # 输出 token 直接作为下一轮的输入，不再拼接
        text = tokenizer.decode(in_tokens)
        print(f'step {i} input: {text}', flush=True)
        i += 1
        out_text += text # 拼接当前输出 token 解码对应的文本  

print(f' Input: {in_text}')
print(f'Output: {out_text}')
```
## gpt2  https://huggingface.co/docs/transformers/model_doc/gpt2    

```
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path='bert-base-chinese',
    cache_dir=None,
    force_download=False,
)

out = tokenizer.encode_plus(
    text="我喜欢大语言模型",
    text_pair="因为它改变了自然语言处理",
    padding='max_length',
    max_length=20,
    add_special_tokens=True,
    return_tensors=None,
    return_token_type_ids=True,
    return_attention_mask=True,
    return_special_tokens_mask=True,
)

for k, v in out.items():
    print(k, ':', v)

tokenizer.decode(out['input_ids'])

输出：
input_ids : [101, 2769, 1599, 3614, 1920, 6427, 6241, 3563, 1798, 102, 1728, 711, 2124, 3121, 1359, 749, 5632, 4197, 6427, 6241, 1905, 4415, 102, 0, 0, 0, 0, 0, 0, 0]
token_type_ids : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
attention_mask : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

[CLS] 我 喜 欢 大 语 言 模 型 [SEP] 因 为 它 改 变 了 自 然 语 言 处 理 [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
```

```
from transformers import GPT2LMHeadModel

# 该路径为本地路径
name_or_path = 'pre_trained/gpt'
# 会自动加载name_or_path中的config.json, pytorch_model.bin
lm_gpt2 = GPT2LMHeadModel.from_pretrained(name_or_path)

bs = 16;_len = 40
x = torch.randn(size=(bs, _len))
outputs = lm_gpt2(input_ids=x,
                  token_type_ids=None,  
                  position_ids=None,  
                  attention_mask=None )
```

https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L753C1-L768C6   

### GPT2LMHeadModel 
```
"""
The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
embeddings).
"""
class GPT2LMHeadModel(GPT2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        # config see https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/configuration_gpt2.py   
        self.transformer = GPT2Model(config) 

        # 分类器  
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    ...

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,                 # (bs, len); sequence 经过 tokenizer 编码后得到的张量
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None, # 第一次为None，GPT2Model会进行扩展, 后续使用kv cache    
        attention_mask: Optional[torch.FloatTensor] = None,           # (bs, len) 标记 input_ids 中 pad 的位置  
        token_type_ids: Optional[torch.LongTensor] = None,            # (bs, len) 区分 input_ids 中第一个 sequence 跟第二个 sequence   
        position_ids: Optional[torch.LongTensor] = None,              # (bs, len) 默认使用绝对位置编码，其范围为[0, config.max_position_embeddings - 1], GPT2Model检测到position_ids为None，会自己创建
        head_mask: Optional[torch.FloatTensor] = None,              # (num_heads,)` or `(num_layers, num_heads)   
        inputs_embeds: Optional[torch.FloatTensor] = None,          # (bs, len, hidden_size)，注意：inputs_embeds和input_ids只能选择一个输入
        encoder_hidden_states: Optional[torch.Tensor] = None,       # 默认None，用于交叉注意力，需要设置config.json，添加add_cross_attention: true
同时需要给值  
        encoder_attention_mask: Optional[torch.FloatTensor] = None, # 默认None，用于交叉注意力，需要设置config.json，添加add_cross_attention: true
如果add_cross_attention为True，encoder_attention_mask用户不设置的话，程序认为全部有效   
        labels: Optional[torch.LongTensor] = None,  # (bs, len) input_ids  
        use_cache: Optional[bool] = None,           # True 和 past_key_values搭配使用  
        output_attentions: Optional[bool] = None,   # 默认False，是否需要输出GPT2Model处理过程中每个block的attentions
        output_hidden_states: Optional[bool] = None,# 默认False，是否需要输出GPT2Model中每个GPT2Block的输出hidden states
        return_dict: Optional[bool] = None,         # 默认是True，会返回一个结果对象，可以通过.属性值方式获取数据；如果设置为False，则会将需要输出的结果转为元组
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        # 使用config 或下发  
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # GPT2Model最后一个Block的输出   
        hidden_states = transformer_outputs[0]

        # 进行 nn.Linear  
        lm_logits = self.lm_head(hidden_states)

        # 计算 loss  
        # 一般取 input_ids, shape [batch_size, sequence_len], 第一个 token 是 [CLS]，计算损失时要去除
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)

            # Shift so that tokens < n predict n
            # lm_logits shape [batch, sequence_len - 1, vocab_size], [,-1:,:] 是本次计算预测的结果
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # 交叉熵  Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # 默认 return_dict True         
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
```

### GPT2Model    
```
# model_type = "gpt2"
# keys_to_ignore_at_inference = ["past_key_values"]
# attribute_map = {
#     "hidden_size": "n_embd",
#     "max_position_embeddings": "n_positions",
#     "num_attention_heads": "n_head",
#     "num_hidden_layers": "n_layer",
# }

class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)

        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)

        # block 堆叠, 默认num 12 
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])

        # ln 处理  
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        ...

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,# input_ids与inputs_embeds 不可同时设置非None，但必须设置一个非None
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None, # 默认 True   
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    )

  # past_key_values，layer_past      
  if past_key_values is None:  # 第一次infer时，该值为None         
      past_length = 0
      past_key_values = tuple([None] * len(self.h)) # 使用None初始化,扩展长度为num_hidden_layers，即block num 
  else:
      past_length = past_key_values[0][0].size(-2)
  ...

  # 词向量编码
  if inputs_embeds is None:
      inputs_embeds = self.wte(input_ids)

  # 位置编码
  position_embeds = self.wpe(position_ids)
  hidden_states = inputs_embeds + position_embeds

  if token_type_ids is not None:
    token_type_embeds = self.wte(token_type_ids)
    hidden_states = hidden_states + token_type_embeds
    
  # 此时初始状态是 inputs_embeds + position_embeds + token_type_embeds
  # dropout 泛化 
  hidden_states = self.drop(hidden_states)

  # 下面四个tuple 保存中间结果
  presents = () if use_cache else None # 第一次infer presents为空tuple 
  all_self_attentions = () if output_attentions else None
  all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
  all_hidden_states = () if output_hidden_states else None

  # 注意：self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
  for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

        # 还没有进入block处理的hidden states也被保存了
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = block(
            hidden_states,
            layer_past=layer_past, # layer_past 实际就是 past_key_values 的迭代项   
            attention_mask=attention_mask,
            head_mask=head_mask[i],
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        # 注意：hidden_states 被更新，下一次block循环会使用  outputs = hidden_states, present
        hidden_states = outputs[0]

        # outputs[1] 代表key value，每一个block 的attention 中key value 被存储 
        if use_cache is True:
            presents = presents + (outputs[1],)
        
        # 每一个block的 self-attention 中的输出attention 被存储 
        if output_attentions:
            all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
            if self.config.add_cross_attention:
                all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)
        ...

  # All blocks finished ，对最后一个block的输出 hidden_states(outputs[0]) 进行ln处理     
  hidden_states = self.ln_f(hidden_states).view(output_shape)

  # Add last hidden state 此时的hidden state(已经过了ln处理)
  if output_hidden_states:
      all_hidden_states = all_hidden_states + (hidden_states,)

  # 默认使用 dict  
  if not return_dict:
      return tuple(
          v
          for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
          if v is not None
      )

  return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
        cross_attentions=all_cross_attentions,
  )
```
GPT2Model output:   
(hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions)     

### GPT2Attention        
```    
class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim: # 必须要求 embed_dim 可以 整除 num_heads
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights #True
        self.is_cross_attention = is_cross_attention

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim) 
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Q*(K`T) (make K transpose) 
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        # 1/sqrt(n_embd) 
        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        # 执行 softmax   
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)

        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # attn_weights*V
        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

    # (batch, sequence_len, embeded_dim) -> (batch, num_heads, sequence_len, attn_head_size)
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)    # -> (batch, sequence_len, num_heads, attn_head_size)
        return tensor.permute(0, 2, 1, 3)  # -> (batch, num_heads, sequence_len, attn_head_size)
    
    # (batch, num_heads, sequence_len, attn_head_size) -> (batch, sequence_len, embed_dim)
    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous() # -> (batch, sequence_len, num_heads, attn_head_size)
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,) # -> (batch, sequence_len, embed_dim)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        # 如果使用 cross atten  （self.config.add_cross_attention）
        if encoder_hidden_states is not None:
            query = self.q_attn(hidden_states) # hidden_states 来自输入
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2) 
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        # 按num_heads 数据拆分重排, 没有运算  
        query = self._split_heads(query, self.num_heads, self.head_dim)  
        key = self._split_heads(key, self.num_heads, self.head_dim)   
        value = self._split_heads(value, self.num_heads, self.head_dim)  

        # 当输出第一个token后，layer_past就是非None了
        if layer_past is not None:
            past_key, past_value = layer_past # 使用历史 token 对应的kv 
            key = torch.cat((past_key, key), dim=-2) # 拼接历史和当前 token 对应的k
            value = torch.cat((past_value, value), dim=-2) # 拼接历史和当前 token 对应的v 

        if use_cache is True:
            present = (key, value) # 备份key, value, 随着 attn_output 一起输出  
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # 合并head 数据重排, 没有运算
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

        attn_output = self.c_proj(attn_output)

        # dropout infer阶段无
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present) # present 在此处作为outputs的一部分输出  
        if output_attentions:
            outputs += (attn_weights,)

        # attentions(attn_weights) 作为可选输出  attn_output, present, attn_weights
        return outputs  # a, present, (attentions)
```   
还需要注意 kv cache在调用侧的设置（右侧有cache使用）    
![kv-cache](https://github.com/lix19937/history/assets/38753233/10e55e49-689a-415b-9922-776dd4337284)

## kv cache 数学原理   

