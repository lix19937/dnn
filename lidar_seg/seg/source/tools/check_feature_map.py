import numpy as np
import cv2

fm_cpp = np.fromfile('/home/igs/Downloads/feature.bin',
                     dtype=np.float32).reshape(-1, 1024, 512)
fm_python = np.fromfile(
    '/home/igs/catkin_ws/src/nn-detection/source/1.bin', dtype=np.float32).reshape(-1, 1024, 512)

for i in range(16):
    cv2.imshow('py', fm_python[i])
    cv2.waitKey(0)

print(fm_cpp.shape)
print(fm_python.shape)
assert len(fm_python) == len(fm_cpp)

for i in range(len(fm_python)):
    # assert fm_cpp[i] == fm_python[i]
    delta = np.abs(fm_cpp[i] - fm_python[i])

    if delta > 0.0001:
        print('===================')
        print('wrong', delta)
        print('cpp', fm_cpp[i])
        print('python', fm_python[i])
    else:
        pass
        # print('good', delta)
