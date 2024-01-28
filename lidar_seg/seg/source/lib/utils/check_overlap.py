import numpy as np
import time


class CheckOverlap:
    def __init__(self) -> None:
        pass

    def _perp(self, v):
        return np.array([v[1], -v[0]], dtype=np.float32)

    def _minus_points(self, v1, v2):
        return np.array([v1[0] - v2[0], v1[1] - v2[1]], dtype=np.float32)

    def _mul_points(self, v1, v2):
        return v1[0] * v2[0] + v1[1] * v2[1]

    def _projection(self, polygon, v):
        min_tmp = self._mul_points(polygon[0], v)
        max_tmp = min_tmp
        for i in range(len(polygon)):
            p = self._mul_points(polygon[i], v)
            min_tmp = min(min_tmp, p)
            max_tmp = max(max_tmp, p)
        return np.array([min_tmp, max_tmp], dtype=np.float32)

    def _get_edges_from_vertices(self, polygon):
        edges = []
        for i in range(len(polygon) - 1):
            edges.append(self._minus_points(polygon[i + 1], polygon[i]))
        edges.append(self._minus_points(polygon[0], polygon[-1]))
        return edges

    def _contains(self, val, vec):
        return val > vec[0] and val < vec[1]

    def _is_overlap(self, v1, v2):
        return (self._contains(v1[0], v2) or self._contains(v1[1], v2) or
                self._contains(v2[0], v1) or self._contains(v2[1], v1))

    def is_overlap(self, polygon1, polygon2):
        if len(polygon1) < 3 or len(polygon2) < 3:
            return False
        edges = []
        edges.extend(self._get_edges_from_vertices(polygon1))
        edges.extend(self._get_edges_from_vertices(polygon2))
        for i in range(len(edges)):
            direction = edges[i]
            direction = self._perp(direction)
            proj1 = self._projection(polygon1, direction)
            proj2 = self._projection(polygon2, direction)
            if not self._is_overlap(proj1, proj2):
                return False
        return True
