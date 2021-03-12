import random
import math

import numpy as np
import config_sys
config_sys.set_sys_path()

from utils import polar2xy, xy2polar, get_from_dict, search_dict

def get_polygon_regular(**kw):
    edge_num = search_dict(kw, ["point_num", "edge_num"])
    square_max_size = search_dict(kw, ["box_width", "edge_length", "box_size", "square_max_size", "square_size"])
    direct_offset = search_dict(kw, ["direct", "direct_offset"])
    center_coord = search_dict(kw, ["center", "center_coord"], default=[0.0, 0.0])
    center_x, center_y = center_coord[0], center_coord[1]

    r = square_max_size / 2
    theta_delta = math.pi * 2 / edge_num
    points = np.zeros([edge_num, 2])
    if direct_offset is None:
        if random_direct:
            theta_now = random.random() * theta_delta # srandom.random() generate a random float in (0, 1)
        else:
            theta_now = 0.0
    else:
        theta_now = direct_offset
    for num in range(edge_num):
        x, y = polar2xy(r, theta_now)
        points[num, 0], points[num, 1] = x + center_x, y + center_y
        theta_now += theta_delta
    return points
def get_polygon_vecs(points):
    edge_num = points.shape[0]       
    vecs = np.zeros([edge_num, 2])
    vecs[0:edge_num-1, :] = np.diff(points, axis=0)
    vecs[-1, :] = points[0, :] - points[-1, :] #the last side points from the last point to the first point.
    return vecs
def get_polygon_norms(points, vecs): #get norm vectors pointing outside polygon.
    edge_num = points.shape[0]  
    norms = np.zeros([edge_num, 2])
    #(a, b) is vertical to (b, -a)
    norms[:, 0] = vecs[:, 1]
    norms[:, 1] = vecs[:, 0] * (-1.0)
    
    norms = norms / (np.linalg.norm(norms, ord=2, axis=-1))[:, np.newaxis]
    norms_angle = np.zeros([edge_num])
    for num in range(edge_num):
        #print("x, y: %f, %f"%(norms[num, 0], norms[num, 1]))
        r, theta = xy2polar(norms[num, 0], norms[num, 1])
        #print("r, theta: %f, %f"%(r, theta))
        norms_angle[num] = theta
    #print(norms)
    #print(norms_angle)
    return norms, norms_angle
def get_dist_to_edges(coords, vertices, norms):
    dist_vecs = vertices[np.newaxis, :, :] - coords[:, np.newaxis, :] #[1, vertex_num, 2] - [point_num, 1, 2] = [point_num, vertex_num, 2]
    return np.sum(dist_vecs * norms[np.newaxis, :, :], axis=-1) #[point_num, vertex_num]