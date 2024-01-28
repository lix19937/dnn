import os
import numpy as np
# import rospy
# from sensor_msgs.msg import PointCloud2, PointField
# from visualization_msgs.msg import MarkerArray, Marker
# from geometry_msgs.msg import Point
# import cv2
from sklearn.cluster import DBSCAN
from scipy.spatial import Delaunay
from tqdm import tqdm

def is_in_poly(p, poly):
    """
    :param p: [x, y]
    :param poly: [[], [], [], [], ...]
    :return:
    """
    px, py = p[0], p[1]
    is_in = False
    for i, corner in enumerate(poly):
        next_i = i + 1 if i + 1 < len(poly) else 0
        x1, y1 = corner[0], corner[1]
        x2, y2 = poly[next_i][0], poly[next_i][1]
        if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x == px:  # if point is on edge
                is_in = True
                break
            elif x > px:  # if point is on left-side of line
                is_in = not is_in
    return is_in

def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)

    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle

    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)

        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges

def generate_poly(corners):
    start = list(corners.values())
    t = [_ for _ in start if len(_) > 1]
    start = t[0][0][0] if len(t) > 0 else start[0][0][0]
    # start = [_ for _ in start if len(_) > 1][0][0][0]
    vertices = []
    while len(corners) > 0:
        vertices.append(start)
        if start not in corners:
            vertices.pop(-1)
            if len(vertices) < len(corners):
                vertices = []
            else:
                return vertices
            start = list(corners.values())
            t = [_ for _ in start if len(_) > 1]
            start = t[0][0][0] if len(t) > 0 else start[0][0][0]
        item = corners[start]
        _, start = item[0]
        item.pop(0)
        if len(item) == 0:
            corners.pop(_)
    return vertices

def change_8_6(pc, label, polygon, param, param_d):
    filter_idx = label == 8
    label_idx = np.linspace(0, pc.shape[0] - 1, pc.shape[0], dtype=np.int32)
    label_idx = label_idx[filter_idx]
    points = pc[filter_idx, :]
    if points.shape[0] == 0:
        return False
    
    ##设置分层聚类函数
    db = DBSCAN(eps=1, min_samples=5)
    ##训练数据
    db.fit(points)
    cluster_num = np.unique(db.labels_)
    changed = False
    for id in cluster_num[1:]:
        idx = db.labels_ == id
        pts = points[idx, :]
        in_poly_idx = []
        for i, pt in enumerate(pts):
            if is_in_poly(pt, polygon):
                in_poly_idx.append(i)
        if len(in_poly_idx) == 0:
            continue
        pts = pts[in_poly_idx, :]
        ground_z = np.matmul(pts, param.reshape(3, 1)) + param_d
        if np.min(ground_z) < 1.8:
            continue
        lbs = label_idx[idx]
        label[lbs] = 6
        print('change ', id, 'cluster to suspension!!')
        changed = True
    return changed

def change_6_8(pc, label, polygon, param, param_d):
    filter_idx = label == 6
    label_idx = np.linspace(0, pc.shape[0] - 1, pc.shape[0], dtype=np.int32)
    label_idx = label_idx[filter_idx]
    points = pc[filter_idx, :]
    
    ##设置分层聚类函数
    db = DBSCAN(eps=1, min_samples=5)
    ##训练数据
    db.fit(points)
    cluster_num = np.unique(db.labels_)
    changed = False
    for id in cluster_num[1:]:
        idx = db.labels_ == id
        pts = points[idx, :]
        in_polygon = False
        for pt in enumerate(pts):
            if is_in_poly(pt, polygon):
                in_polygon = True
                break
        if in_polygon:
            continue
        ground_z = np.matmul(pts, param.reshape(3, 1)) + param_d
        if np.min(ground_z) < 1.8:
            continue
        lbs = label_idx[idx]
        label[lbs] = 6
        print('change ', id, 'cluster to suspension!!')
        changed = True
    return changed

data_dir = '/data/luminar_seg/aligned_seg/single/sunny/'
data_paths = os.listdir(data_dir)

# rospy.init_node('talker_p')
# pub =rospy.Publisher('pc', PointCloud2, queue_size=10)
# msg = PointCloud2()
# msg.header.stamp = rospy.Time().now()
# msg.header.frame_id = "base_link"
    
# msg.fields = [
#     PointField('x', 0, PointField.FLOAT32, 1),
#     PointField('y', 4, PointField.FLOAT32, 1),
#     PointField('z', 8, PointField.FLOAT32, 1),
#     PointField('label', 12, PointField.FLOAT32, 1)]
# msg.is_bigendian = False
# msg.point_step = 16
# msg.height = 1

# pub_marker=rospy.Publisher('box', MarkerArray, queue_size=10)

file = open('./change_label.txt', 'a')
for task in data_paths:
    task_path = os.path.join(data_dir, task, 'train.txt')
    if not os.path.exists(task_path):
        continue
    pcs = open(task_path, 'r').readlines()
    pcs = [_.strip() for _ in pcs]
    # a = pcs.index(now)
    for data_path in tqdm(pcs):
        # data_path = data_path.strip()
        pc = np.fromfile(data_path, np.float32).reshape(-1, 3)
        label_path = data_path.replace('pcd_dir', 'ann_dir')
        label = np.fromfile(label_path, np.uint8)
        grounds = pc[label == 0, :]
        if (grounds.shape[0] < 4):
            continue
        hull = alpha_shape(grounds[:, :2], alpha=20)
        corners = dict()
        for t in hull:
            if t[0] in corners:
                corners[t[0]].append(t)
            else:
                corners[t[0]] = [t]
        vertices = generate_poly(corners)
        corners = grounds[vertices, :]
        
        center = np.mean(grounds, axis=0).reshape(1, -1)
        aligned_pts = grounds - center
        a,b = np.linalg.eig(np.matmul(aligned_pts.T, aligned_pts))
        vec = b[:, np.argmin(a)]
        param_D = -np.dot(vec, center.reshape(-1))
        
        if change_8_6(pc, label, corners, vec, param_D):
            label.tofile(label_path)
            file.write(data_path + '\n')
            file.flush()
        
        # msg_maker = MarkerArray()
        # marker = Marker()
        # marker.type = Marker.LINE_LIST
        # marker.header.frame_id = 'base_link'
        # marker.header.stamp = rospy.Time().now()
        # marker.ns = 'bbox'
        # marker.id = 0
        # marker.pose.orientation.w = 1
        # marker.action = Marker.ADD
        # marker.color.a = 1
        # marker.color.r = 1
        # marker.color.g = 1
        # marker.color.b = 1
        # marker.scale.x = 0.2
        # # t1 = np.array([box[0], 0, 0, 0, box[1], 0, 0, 0, box[2]], np.float).reshape(3, -1)
        # # crs=[min_p, max_p - t1[2,:], max_p - t1[0, :], max_p - t1[1, :]]
        # # crs = [np.dot(_, b.T) for _ in crs]
        # # crs1=[max_p, min_p + t1[2,:], min_p + t1[0, :], min_p + t1[1, :]]
        # # crs1 = [np.dot(_, b.T) for _ in crs1]
        # # pts=[max_p, max_p + t1, max_p + t1 + t2, max_p + t2,max_p]
        # for i in range(corners.shape[0]):
        #     next_i = i + 1 if i + 1 < corners.shape[0] else 0
        #     p =Point()
        #     p.x = corners[i][0]
        #     p.y = corners[i][1]
        #     p.z = corners[i][2]
        #     marker.points.append(p)
        #     p =Point()
        #     p.x = corners[next_i][0]
        #     p.y = corners[next_i][1]
        #     p.z = corners[next_i][2]
        #     marker.points.append(p)
        # msg_maker.markers.append(marker)
        
        # # print()
        # points = np.concatenate((pc, label.reshape(-1, 1)), axis=1)
        # msg.row_step = msg.point_step * points.shape[0]
        # msg.width = points.shape[0]
        # msg.is_dense = False
        # msg.data = np.asarray(points, np.float32).tostring()
        # pub.publish(msg)
        # pub_marker.publish(msg_maker)

        # cv2.imshow('img', img)
        # cv2.imshow('rv', rv/20)
        # if cv2.waitKey() == 27:
        #     exit()
file.close()