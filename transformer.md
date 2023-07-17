
* transformers_families  
![transformers_families](https://github.com/lix19937/pytorch-cookbook/assets/38753233/ed572453-f458-4e72-94e9-165e489f9984)

* transformers_blocks   
![transformers_blocks](https://github.com/lix19937/pytorch-cookbook/assets/38753233/0a40f2d5-38f5-4ff2-9a5f-740a84815513)

* transformers_base  
![transformers_base](https://github.com/lix19937/pytorch-cookbook/assets/38753233/3780289b-fe7c-4aaf-afa0-826462410e9d)    


GPT-like (也被称作自回归Transformer模型)   
BERT-like (也被称作自动编码Transformer模型)   
BART/T5-like (也被称作序列到序列的 Transformer模型)      


|type| transformer-block | arch | task | examples  |   
|----|-------------------|------|------|-----------|      
|BERT-like   |  Encoder-only   | 自动编码 | 句子分类、命名实体识别（以及更普遍的单词分类）和阅读理解后回答问题 | ALBERT, BERT, DistilBERT, ELECTRA, RoBERTa  |
|GPT-like   |  Decoder-only   | 自动回归 | 文本生成 | CTRL, GPT, GPT-2,Transformer XL  |
|BART/T5-like   |  Encoder-Decoder   | 序列到序列 | 摘要、翻译或生成性问答 | BART, mBART, Marian, T5  |   


ref https://huggingface.co/learn/nlp-course/chapter1/2?fw=pt   

