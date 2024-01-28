import torch
import preprocess_cpp
import numpy as np
import time

cloud = np.fromfile(
    '/home/igs/Downloads/debug_data/clouds/0001/bin/1632793372938819000.bin', dtype=np.float32)
cloud = cloud.reshape(-1, 3)
print('cloud before aug', cloud)

t0 = time.time()

cloud = torch.from_numpy(cloud)
fm = preprocess_cpp.build(cloud, 16, 512, 512, -30.0,
                          -51.2, -1.2, 3.0, 0.2, 0.3)
fm = fm.numpy()

t1 = time.time()
print('time', t1 - t0)

t2 = time.time()
cloud = preprocess_cpp.augment(
    cloud, np.sin(0.), np.cos(0.), 0.5, 0.5, 0.1)
t3 = time.time()
print('aug time', t3 - t2)
print('cloud after aug', cloud.numpy())


print(fm.shape)
print(np.argwhere(fm > 1.0))
print(fm[15, 509, 113])
