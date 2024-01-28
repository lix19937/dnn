from grid_map import GridMap
import numpy as np

def bresenham(start, end):
    ans = []
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    ux = 1 if dx > 0 else -1
    uy = 1 if dy > 0 else -1
    dx = abs(dx)
    dy = abs(dy)
    dx2 = dx << 1
    dy2 = dy << 1
    x = start[0]
    y = start[1]
    if dx > dy:
        e = -dx
        while x != end[0]:
            ans.append((x, y))
            e += dy2
            if e > 0:
                y += uy
                e -= dx2
            x += ux
    else:
        e = -dy
        while y != end[1]:
            ans.append((x, y))
            e += dx2
            if e > 0:
                x += ux
                e -= dy2
            y += uy
    ans.append((end[0], end[1]))
    return ans

class ExtractFs:
    def __init__(self) -> None:
        self.grid_map = GridMap()
        self.row = self.grid_map.row
        self.col = self.grid_map.col
        self.ob_label = [1, 2, 3, 4, 8, 11]
        self.search_path = []
        self.init_search_path()

    def init_search_path(self):
        row_max = self.row - 1
        col_max = self.col - 1
        steps = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        end_indices = [(row_max, 0), (row_max, col_max), (0, col_max), (0, 0)]
        start_idx = (5, int(self.col / 2))
        idx = [0, 0]
        for i, step in enumerate(steps):
            while idx[0] != end_indices[i][0] or idx[1] != end_indices[i][1]:
                idx[0] += step[0]
                idx[1] += step[1]
                self.search_path.append(bresenham(start_idx, idx))
    
    def build(self, cloud: np.ndarray, label: np.ndarray):
        img = np.zeros((self.row, self.col), np.uint8)
        valid_indices = [idx for idx, _ in enumerate(label) if _ in self.ob_label]
        obstacle = cloud[valid_indices, :]
        valid_label = label[valid_indices]
        local = self.grid_map.build(obstacle)
        for paths in self.search_path:
            for cell in paths:
                if cell not in local:
                    continue
                img[cell] = 255
                for pt_i in local[cell]:
                    valid_label[pt_i] = 18
                break
        return img