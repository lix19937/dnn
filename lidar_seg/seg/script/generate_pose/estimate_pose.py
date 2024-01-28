import numpy as np
import math

from torch import le
from grid_map import GridMap

# 求出两个点之间的向量角度，向量方向由点1指向点2
def getThetaOfTwoPoints(x1, y1, x2, y2):
    return math.atan2(y2-y1, x2-x1)

# 求出两个点的距离
def getDistOfTwoPoints(x1, y1, x2, y2):
    return math.sqrt(math.pow(x2-x1, 2) + math.pow(y2-y1, 2))

# 在pt_set点集中找到距(p_x, p_y)最近点的id
def getClosestID(pc, pt_set):
    id = 0
    min = 10000000
    for i, pc2 in enumerate(pt_set):
        dist = np.linalg.norm(pc - pc2)
        if dist < min:
            id = i
            min = dist
    return id, min

def find_match_pcs(set1, set2, dist_threshold):
    idx1 = []
    idx2 = []
    dist_mean = 0
    for i, pc in enumerate(set1):
        j, dist = getClosestID(pc, set2)
        if dist > dist_threshold:
            continue
        dist_mean += dist
        idx1.append(i)
        idx2.append(j)
    if len(idx1) == 0:
        return None, None, None
    return set1[idx1, :2].T, set2[idx2, :2].T, dist_mean / len(idx1)

# 求出两个点集之间的平均点距
def DistOfTwoSet(set1, set2):
    dist = 0
    for i in range(len(set1)):
        dist += np.linalg.norm(set1[i] - set2[i])
    return dist / len(set1)

# ICP核心代码
def ICP(sourcePoints, targetPoints, init_R, init_t):
    if init_R is not None:
        origin_R = init_R
    else:
        origin_R = np.identity(3, np.float32)
    if init_t is not None:
        origin_t = init_t
    else:
        origin_t = np.zeros((3, 1), np.float32)
    sourcePoints = (origin_R@sourcePoints.T + origin_t).T
    iteration_times = 0
    dist_improve = 1
    dist_before = 5
    dist_now = 0
    while iteration_times < 10 and dist_improve > 0.001:
        B, A, dist_before = find_match_pcs(sourcePoints, targetPoints, dist_before*2)
        if B is None or iteration_times == 0 and B.shape[1] < 4:
            break
        x_mean_target = A[0].mean()
        y_mean_target = A[1].mean()
        x_mean_source = B[0].mean()
        y_mean_source = B[1].mean()

        A_ = A - np.array([[x_mean_target], [y_mean_target]])
        B_ = B - np.array([[x_mean_source], [y_mean_source]])

        w_up = 0
        w_down = 0
        for i in range(A_.shape[1]):
            j = i
            w_up_i = A_[0][i]*B_[1][j] - A_[1][i]*B_[0][j]
            w_down_i = A_[0][i]*B_[0][j] + A_[1][i]*B_[1][j]
            w_up = w_up + w_up_i
            w_down = w_down + w_down_i

        theta = math.atan2(w_up, w_down)
        x = x_mean_target - math.cos(theta)*x_mean_source - math.sin(theta)*y_mean_source
        y = y_mean_target + math.sin(theta)*x_mean_source - math.cos(theta)*y_mean_source
        R = np.array([[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0],
        [0, 0, 1]])

        t = np.array([[x], [y], [0]])
        B = np.matmul(R[:2, :2], B) + t[:2]
        origin_R = np.matmul(origin_R, R)
        origin_t += t

        iteration_times = iteration_times + 1
        dist_now = DistOfTwoSet(A.T, B.T)
        dist_improve = dist_before - dist_now
        # print("迭代第"+str(iteration_times)+"次, 损失是"+str(dist_now)+",提升了"+str(dist_improve))
        dist_before = dist_now
        sourcePoints = (R@sourcePoints.T + t).T

    return origin_R, origin_t, dist_now

def GetGroundParam(grounds):
    center = np.mean(grounds, axis=0).reshape(1, -1)
    aligned_pts = grounds - center
    a,b = np.linalg.eig(np.matmul(aligned_pts.T, aligned_pts))
    vec = b[:, np.argmin(a)]
    param_D = -np.dot(vec, center.reshape(-1))
    return [vec, param_D]

def TLG(sourcePoints, targetPoints):
    src_param, src_z = GetGroundParam(sourcePoints)
    tgt_param, tgt_z = GetGroundParam(targetPoints)
    z = tgt_z - src_z
    sita = math.acos(min(1, tgt_param@src_param))
    if sita > math.pi / 2:
        sita = math.pi - sita
    n_vector = np.cross(tgt_param, src_param)
    
    n_vector = n_vector / np.linalg.norm(n_vector)
    
    n_vector_invert = np.array([
    [0,-n_vector[2],n_vector[1]],
    [n_vector[2],0,-n_vector[0]],
    [-n_vector[1],n_vector[0],0]
    ])
    
    I = np.identity(3, np.float32)
    R_w2c = I + math.sin(sita)*n_vector_invert + n_vector_invert@(n_vector_invert)*(1-math.cos(sita))
    return R_w2c, float(z)

class EstimatePose:
    def __init__(self) -> None:
        self.grid_map = GridMap()
        self.steps = ((0, 1), (1, 0), (0, -1), (-1, 0))

    def GetClusterCenter(self, pc: np.ndarray):
        clusters = self.grid_map.build(pc)
        searched = set(clusters.keys())
        total_cluster = []
        while len(searched) > 0:
            cell = searched.pop()
            que = [cell]
            cluster = []
            while len(que) > 0:
                t_cell = que.pop(0)
                cluster.append(t_cell)
                for step in self.steps:
                    tmp_cell = (t_cell[0] + step[0], t_cell[1] + step[1])
                    if tmp_cell in searched:
                        que.append(tmp_cell)
                        searched.remove(tmp_cell)
            center = np.zeros(3, np.float32)
            size = 0
            for t in cluster:
                t_sum = sum(pc[clusters[t], :])
                size += len(clusters[t])
                center += t_sum
            center /= size
            total_cluster.append(center)
        return np.array(total_cluster)
        