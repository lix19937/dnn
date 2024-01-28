import cv2
import numpy as np

class Hist(dict):
    def __init__(self) -> None:
        super().__init__()

    def __init__(self, idx, label) -> None:
        super().__init__()
        self[label] = [idx]

    def add(self, idx, label) -> None:
        if label not in self:
            self[label] = [idx]
        else:
            self[label].append(idx)

class GridMap:
    def __init__(self) -> None:
        self.res = 0.4
        self.x_min = 0
        self.y_min = -50
        self.row = 375
        self.col = 250
        self.transform = np.eye(3)
        self.transform[0, 2] = -self.x_min
        self.transform[1, 2] = -self.y_min
        self.transform = self.transform / self.res

    def valid(self, idx):
        return 1 <= idx[0] < self.row and 0 <= idx[1] < self.col

    def get_index(self, x, y):
        return int((x - self.x_min) / self.res), int((y - self.y_min) / self.res)

    def get_rowcol(self, dim):
        return dim // self.col, dim % self.col

    def change(self, idx):
        return idx[0] * self.col + idx[1]

    def build(self, cloud: np.ndarray):
        grid_map = dict()
        for i, pt in enumerate(cloud):
            idx = self.get_index(pt[0], pt[1])
            if not self.valid(idx):
                continue
            if idx in grid_map:
                grid_map[idx].append(i)
            else:
                grid_map[idx] = [i]
        return grid_map
