
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3,4,5,6,7'

from loguru import logger
import torch
import threading
 
print(torch.cuda.is_available())
 
ng = torch.cuda.device_count()
print("Devices:%d" %ng)

infos = [torch.cuda.get_device_name(i) for i in range(ng)]
print(infos)

infos = [torch.cuda.get_device_properties(i) for i in range(ng)]
print(infos)
print(torch.cuda.current_device())
# PYTORCH_CUDA_ALLOC_CONF

for i in range(ng):  
  t = torch.cuda.get_device_properties(0).total_memory
  c = torch.cuda.memory_reserved(0)
  a = torch.cuda.memory_allocated(0)
  f = c-a  # free inside cache
  print(t, c, a, f)
  print("====================")

class MyThread(threading.Thread):
    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.idx = threadID
        torch.cuda.set_device('cuda:' + str(self.idx))  

    def run(self):
        # torch.ones((77//2, 1024*1024*1024), dtype = torch.int8, device = 'cuda:' + str(self.idx))# 
        # torch.ones((85198045184), dtype = torch.int8, device = 'cuda:' + str(self.idx))# 
        # torch.ones((83009, 1000*1000), dtype = torch.int8, device = 'cuda:' + str(self.idx))#  81066MiB  / T 81251MiB
        # torch.ones((83020, 1000*1000), dtype = torch.int8, device = 'cuda:' + str(self.idx))#  81078MiB
        # torch.ones((83070, 1000*1000), dtype = torch.int8, device = 'cuda:' + str(self.idx))#  81124MiB
        # torch.ones((83100, 1000*1000), dtype = torch.int8, device = 'cuda:' + str(self.idx))#  81154MiB
        # torch.ones((83170, 1000*1000), dtype = torch.int8, device = 'cuda:' + str(self.idx))#  81220MiB  
        # torch.ones((83190, 1000*1000), dtype = torch.int8, device = 'cuda:' + str(self.idx))#  81240MiB  
        # torch.ones((83199, 1000*1000), dtype = torch.int8, device = 'cuda:' + str(self.idx))#  81248MiB  
        torch.ones((83200//1, 1000*1000), dtype = torch.int8, device = 'cuda:' + str(self.idx))#  81248MiB  

it = 0
while 1:
    # print ("it ",  it)  
    it +=1  
    threads = []
    for i in range(ng):
      thread = MyThread(i)
      threads.append(thread)

    for t in threads:
        t.start()

    for t in threads:
        t.join()
