
+ 给一个 提示词 pt

+ pt 进行 token 化 （先 embedding  ） ptt

+ ptt 进行第一次infer， kv cache (prefill )， 产生一个输出 t_i 

+ ptt + t_i   进入infer    

