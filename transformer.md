
* transformers_families  
![transformers_families](https://github.com/lix19937/pytorch-cookbook/assets/38753233/ed572453-f458-4e72-94e9-165e489f9984)

* transformers_blocks   
![transformers_blocks](https://github.com/lix19937/pytorch-cookbook/assets/38753233/0a40f2d5-38f5-4ff2-9a5f-740a84815513)

* transformers_base  
![transformers_base](https://github.com/lix19937/pytorch-cookbook/assets/38753233/3780289b-fe7c-4aaf-afa0-826462410e9d)    



encoder，用于对输入的sequence进行表示，得到一个很好特征向量。   
decoder，利用encoder得到的特征，以及原始的输入，进行新的sequence的生成。   
encoder、decoder既可以单独使用，又可以再一起使用，因此，基于Transformer的模型可以分为三大类:     
* Encoder-only
* Decoder-only
* Encoder-Decoder

![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+)`编码器`模型指仅使用编码器的Transformer模型。在每个阶段，注意力层都可以获取初始句子中的所有单词。这些模型通常具有“双向”注意力，被称为自编码模型。  
这些模型的预训练通常围绕着以某种方式破坏给定的句子（例如：通过随机遮盖其中的单词），并让模型寻找或重建给定的句子。如BERT中使用的就是两个预训练任务就是Masked language modeling和Next sentence prediction。   
编码器模型最**适合于需要理解完整句子的任务**，例如：句子分类、命名实体识别（以及更普遍的单词分类）和阅读理解后回答问题。   
该系列模型的典型代表有:ALBERT,BERT,DistilBERT,ELECTRA,RoBERTa

![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+)`解码器`模型通常指仅使用解码器的Transformer模型。在每个阶段，对于给定的单词，注意力层**只能获取到句子中位于将要预测单词前面的单词**。预训练任务通常是Next word prediction，这种方式又被称为Causal language modeling。这个Causal就是“因果”的意思，对于decoder，它在训练时是**无法看到全文的，只能看到前面的信息**。这些模型通常被称为自回归模型。   
解码器模型的预训练通常围绕预测句子中的下一个单词进行。这些模型最适合文本生成的任务。    
该系列模型的典型代表有:CTRL,GPT,GPT-2,Transformer XL

![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+)`编码器-解码器`模型（也称为序列到序列模型)同时使用Transformer架构的编码器和解码器两个部分。在每个阶段，编码器的注意力层可以访问初始句子中的所有单词，而解码器的注意力层只能访问位于输入中将要预测单词前面的单词。   
这些模型的预训练可以使用训练编码器或解码器模型的方式来完成，但通常涉及更复杂的内容。例如，T5通过将文本的随机跨度（可以包含多个单词）替换为单个特殊单词来进行预训练，然后目标是预测该掩码单词替换的文本。  
序列到序列模型最适合于围绕根据给定输入生成新句子的任务，如摘要、翻译或生成性问答。    
该系列模型的典型代表有:BART,mBART,Marian,T5

GPT-like (也被称作自回归Transformer模型)   
BERT-like (也被称作自动编码Transformer模型)   
BART/T5-like (也被称作序列到序列的 Transformer模型)      


|type| transformer-block | arch | task | examples  |    
|----|-------------------|------|------|-----------|      
|BERT-like   |  Encoder-only   | 自动编码 | 句子分类、命名实体识别（以及更普遍的单词分类）和阅读理解后回答问题 | ALBERT, BERT, DistilBERT, ELECTRA, RoBERTa  |     
|GPT-like   |  Decoder-only   | 自动回归 | 文本生成 | CTRL, GPT, GPT-2,Transformer XL  |    
|BART/T5-like   |  Encoder-Decoder   | 序列到序列 | 摘要、翻译或生成性问答 | BART, mBART, Marian, T5  |     

    
    
 



ref https://huggingface.co/learn/nlp-course/chapter1/2?fw=pt   

