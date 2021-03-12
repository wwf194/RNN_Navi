import random
import abc # asbtract method

import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils_plot import *
from utils_plot import get_int_coords, get_int_coords_np, get_res_xy, get_float_coords_np
import utils
from utils import get_from_dict, ensure_path, write_dict_info, set_instance_variable, set_dict_variable, set_dict_and_instance_variable
from utils_arena import *

class Arenas():
    def __init__(self, dict_, options=None, load=False):
        if options is not None:
            self.receive_options(options)
        else:
            raise Exception('Arenas: options must not be None.')
        self.dict = dict_
        self.arenas = []
        for arena_dict in self.dict['arena_dicts']:
            self.arenas.append(self.build_arena(arena_dict, options=self.options, load=load))
    
        self.set_current_arena(0)

    def build_arena(self, arena_dict, options, load):
        type_ = arena_dict['type']
        if type_ in ['sqaure', 'polygon', 'square_max', 'rec', 'rectangle', 'rec_max'] or isinstance(type_, int):
            return Arena_Polygon(arena_dict, self.options, load=load)    
        elif type_ in ['circle']:
            return Arena_Circle(arena_dict, self.options, load=load)
        else:
            raise Exception('Arenas: Unsupported arena_type: '+str(type_))
    
    '''
    def plot_arenas_cv(self, save=True, save_path='./', **kw):
        for index, arena in enumerate(self.arenas):
            arena.plot_arena_cv(save=save, save_path=save_path, save_name='arena_%d_plot.png'%(index), **kw)
    '''
    def plot_arenas(self, save=True, save_path='./', **kw):
        for index, arena in enumerate(self.arenas):
            arena.plot_arena_plt(save=save, save_path=save_path, save_name='arena_%d_plot.png'%(index), **kw)

    def plot_random_xy_cv(self, res=50, save=True, save_path='./', **kw):
        for index, arena in enumerate(self.arenas):
            arena.plot_random_xy_cv(save=save, save_path=save_path, save_name='arena_%d_random_xy.png'%(index), **kw)

    def plot_random_xy(self, res=50, save=True, save_path='./', **kw):
        for index, arena in enumerate(self.arenas):
            arena.plot_random_xy_plt(save=save, save_path=save_path, save_name='arena_%d_random_xy.png'%(index), **kw)

    def receive_options(self, options):
        self.options = options

    def set_current_arena(self, index):
        self.arena_current = self.arenas[index]

    def get_current_arena(self):
        return self.arena_current
    
    def current_arena(self):
        return self.arena_current   

    def get_arena(self, index):
        return self.arenas[index]

class Arena(abc.ABC):
    def __init__(self, dict_, options=None, load=False):
        self.dict = dict_

        
    #@abc.abstractmethod
    def get_random_xy(self): # must be implemented by child class.
        return
    def get_random_max_rectangle(self, point_num):
        return np.stack( [np.random.uniform(self.x0, self.x1, point_num), np.random.uniform(self.y0, self.y1, point_num)], axis=0 ) #[points_num, (x, y)]
    def get_random_xy_max(self, num=100):
        xs = np.random.uniform(self.x0, self.x1, num) # x_0
        ys = np.random.uniform(self.y0, self.y1, num) # y_0
        xys = np.stack([xs,ys], axis=1) # np.stack will insert a dimension at designated axis. [num, 2]
        #print('get_random_xy_max: points shape:'+str(xys.shape))
        return xys
    def save(self, dir_):
        with open(dir_, 'wb') as f:
            net = self.to(torch.device('cpu'))
            torch.save(net.dict, f)
            net = self.to(self.device)
    def get_mask(self, res=None, res_x=None, res_y=None, points_grid=None):
        #print('res_x: %d res_y: %d'%(res_x, res_y))
        if res is not None:
            res_x, res_y = get_res_xy(res, self.width, self.height)
        if points_grid is not None:
            if points_grid.shape[0] != res_x * res_y:
                raise Exception('Arena.get_mask: points_grid shape must be consistent with res_x, res_y.')
        else:
            points_grid_int = np.empty(shape=[res_x, res_y, 2], dtype=np.int) # coord
            for i in range(res_x):
                points_grid_int[i, :, 1] = i # y_index
            for i in range(res_y):
                points_grid_int[:, i, 0] = i # x_index
            points_grid = get_float_coords_np( points_grid_int.reshape(res_x * res_y, 2), self.xy_range, res_x, res_y ) # [res_x * res_y, 2]
        



        arena_mask = np.ones((res_x, res_y)).astype(np.bool)
        #print('res_x: %d res_y: %d'%(res_x, res_y))
        #print(points_grid.shape)
        points_out_of_region = self.out_of_region(points_grid, thres=0.0)
        if points_out_of_region.shape[0] > 0:
            print(points_out_of_region)
            print(points_out_of_region.shape)
            points_out_of_region = np.array([points_out_of_region//res_x, points_out_of_region%res_x], dtype=np.int)
            print(points_out_of_region)
            print(points_out_of_region.shape)           
            points_out_of_region = points_out_of_region.transpose((1,0))
            
            for i in range(points_out_of_region.shape[0]):
                arena_mask[points_out_of_region[i,0], points_out_of_region[i,1]] = 0.0        
            #arena_mask[points_out_of_region] = 0.0 does not work.
        return arena_mask

class Arena_Polygon(Arena):
    def __init__(self, dict_, options=None, load=False):
        super().__init__(dict_, options, load)
        
        set_instance_variable(self, self.dict, ['width', 'height', 'type'])
        #self.width = self.dict['width'] # maximum rectangle
        #self.height =self.dict['height']
        #self.type_ = self.type = self.dict['type']
        
        self.center_coord = get_from_dict(self.dict, 'center_coord', default=[0.0, 0.0], write_default=True)
        # By opencv convention, origin is at the top-left corner of the pircture.
        self.x0 = self.center_coord[0] - self.width / 2
        self.x1 = self.center_coord[0] + self.width / 2
        self.y0 = self.center_coord[1] - self.width / 2
        self.y1 = self.center_coord[1] + self.width / 2
        self.xy_range = (self.x0, self.x0, self.x1, self.y1)
        self.square_min_size = min(self.width, self.height)

        self.get_random_max = self.get_random_square = self.get_random_max_rectangle

        self.edge_num = get_from_dict(self.dict, 'edge_num', default=None, write_default=True)
        # set self.edge_num
        if self.edge_num is None:
            if self.type in ['square', 'rectangle', 'square_max', 'rec_max']:
                self.edge_num = self.dict['edge_num'] = 4
                self.get_random_xy = self.get_random_xy_max
            elif isinstance(self.type, int):
                self.edge_num = self.dict['edge_num'] = self.type_
                self.get_random_xy = self.get_random_xy_polygon
            else:
                raise Exception('Arena_Polygon: Cannot calculate edge_num.')
        
        self.border_region_width = search_dict(self.dict, ['border_region_width'], default=0.03 * self.square_min_size, write_default=True)

        # standardize arena_type str.
        #print(self.type)
        if self.type in ['rec_max', 'square_max']:
            #print('ccc')
            vertices = np.array([[self.x0, self.y0], [self.x1, self.y0], [self.x1, self.y1], [self.x0, self.y1]])
        else:
            #print('ddd')
            self.rotate = get_from_dict(self.dict, 'rotate', default=0.0, write_default=True)
            vertices = get_polygon_regular(edge_num=self.edge_num, square_size=self.square_min_size, direct_offset=self.rotate, center_coord=self.center_coord)
            self.type = self.dict['type'] = 'polygon'
        edge_vecs = get_polygon_vecs(vertices)
        edge_norms, edge_norms_theta = get_polygon_norms(vertices, edge_vecs)

        set_dict_and_instance_variable(self, self.dict, locals(), keys=['vertices', 'edge_vecs', 'edge_norms', 'edge_norms_theta'])
        
        self.out_of_region = self.out_of_region_polygon
        self.avoid_border = self.avoid_border_polygon
        self.plot_arena = self.plot_arena_plt
        #self.plot_arena = self.plot_arena_plt
        #self.plot_arena(save_path='./anal/')
        #self.print_info()
    def print_info(self):
        print('Arena_Polygon: edge_num:%d'%(self.edge_num))
        print('center_coord:(%.1f, %.1f)'%(self.center_coord[0], self.center_coord[1]))
        print('vertices:', end='')
        for index in range(self.edge_num):
            print('(%.1f, %.1f)'%(self.vertices[index][0], self.vertices[index][1]), end='')
        print('\n')

    def get_nearest_border(self, points, ):
        return
    def get_random_xy_polygon(self, num=100, thres=None):
        if thres is None or thres in ['default']:
            thres = self.border_region_width
        count = 0
        points_list = []
        while(count < num):
            points = self.get_random_xy_max(int(2.2*num))            
            points = np.delete(points, self.out_of_region(points, thres=thres), axis=0)
            points_list.append(points)
            count += points.shape[0]
            #print('total valid points:%d'%count)
        points = np.concatenate(points_list, axis=0)
        if count > num:
            #print(points.shape)
            points = np.delete(points, random.sample(range(count), count - num), axis=0)
            #print(points.shape)
        return points
    def out_of_region_polygon(self, points, thres=None): # coords:[point_num, (x, y)]. This method is only valid for convex polygon.
        if thres is None or thres in ['default']:
            thres = self.border_region_width
        dists = get_dist_to_edges(points, self.vertices, self.edge_norms) # [point_num, edge_num]
        return np.argwhere( np.sum((dists<thres).astype(np.float), axis=1)>0.0 ).squeeze() # indices of out-of-region points: [out_of_region_point_num]
    def avoid_border_polygon(self, xy, theta, thres=None): # theta: velocity direction.
        if thres is None or thres in ['default']:
            thres = self.border_region_width
        dists = get_dist_to_edges(xy, self.vertices, self.edge_norms) #(batch_size, edge_num)
        dist_nearest = np.min(dists, axis=1) # distance to nearst wall. shape:(batch_size)
        norms_theta = self.edge_norms_theta # [edge_num], range:(-pi, pi)
        nearst_theta = norms_theta[np.argmin(dists, axis=1)] # (batch_size). Here index is tuple.
        
        theta = np.mod(theta, 2*np.pi) # range:(0, 2*pi)
        theta_offset = theta - nearst_theta
        theta_offset = np.mod(theta_offset + np.pi, 2*np.pi) - np.pi # range:(-pi, pi)
        
        towards_wall = np.abs(theta_offset) < np.pi/2
        is_near_wall = (dist_nearest < thres) * towards_wall # [batch_size]
        
        theta_adjust = np.zeros_like(theta) # zero arrays with the same shape of theta.
        theta_adjust[is_near_wall] = np.sign(theta_offset[is_near_wall]) * ( np.pi/2 - np.abs(theta_offset[is_near_wall]) ) #turn to direction parallel to wall
        
        is_near_vertex = ( np.min( np.linalg.norm( self.vertices[np.newaxis, :, :] - xy[:, np.newaxis, :], ord=2, axis=-1), axis=-1 ) < thres ) * towards_wall
        theta_adjust[is_near_vertex] = np.pi # turn around
        return is_near_wall, theta_adjust

    def avoid_border_free(self, position, theta, arena_dict=None):
        is_near_wall = ( np.zeros_like(theta) > 1.0 )
        theta_adjust = np.zeros_like(theta)
        return is_near_wall, theta_adjust

    def avoid_border_square(self, position, hd, box_width=None, box_height=None, arena_dict=None, thres=0.0):
        x = position[:,0]
        y = position[:,1]
        dists = [self.x1 - x, self.y1 - y, x - self.x0, y - self.y0] #distance to all edges (right, top, left, down)
        d_wall = np.min(dists, axis=0) #distance to nearst wall
        angles = np.arange(4)*np.pi/2 
        theta = angles[np.argmin(dists, axis=0)]
        hd = np.mod(hd, 2*np.pi)
        a_wall = hd - theta
        a_wall = np.mod(a_wall + np.pi, 2*np.pi) - np.pi
        
        is_near_wall = (d_wall < thres) * (np.abs(a_wall) < np.pi/2)
        theta_adjust = np.zeros_like(hd) #zero arrays with the same shape of hd.
        theta_adjust[is_near_wall] = np.sign(a_wall[is_near_wall])*(np.pi/2 - np.abs(a_wall[is_near_wall]))
        return is_near_wall, theta_adjust
    def plot_border(self, img): # img: [H, W, C], np.uint8. 
        return # to be implemented
    def plot_arena_cv(self, img=None, line_color=(0,0,0), line_width=1, line_type=4, save=False, save_path='./', save_name='arena_plot.png', **kw):# line_color: (b, g, r)
        if img is None:
            res = search_dict(kw, ['res, resolution'], default=100)
            res_x, res_y = get_res_xy(res, self.width, self.height)
            img = np.zeros([res_x, res_y, 3], dtype=np.uint8)
            img[:,:,:] = (255, 255, 255)

        vertices, width, height = self.dict['vertices'], self.width, self.height
        vertex_num = vertices.shape[0]
        
        #print('xy_range:%s'%str(xy_range))
        
        for i in range(vertex_num):
            vertex_0 = get_int_coords(vertices[i, 0], vertices[i, 1], self.xy_range, res_x, res_y)
            vertex_1 = get_int_coords(vertices[(i+1)%vertex_num, 0], vertices[(i+1)%vertex_num, 1], self.xy_range, res_x, res_y)
            #print('plot line: (%d, %d) to (%d, %d)'%(vertex_0[0], vertex_0[1], vertex_1[0], vertex_1[1]))
            cv.line(img, vertex_0, vertex_1, line_color, line_width, line_type) # line_width seems to be in unit of pixel.
        
        if save:
            ensure_path(save_path)
            #if save_name is None:
            #    save_name = 'arena_plot.png'
            cv.imwrite(save_path + save_name, img[:, ::-1, :]) # so that origin is in left-bottom corner.
        return img
    
    def plot_arena_plt(self, ax=None, save=True, save_path='./', save_name='arena_random_xy.png', line_width=2, color=(0,0,0)):
        if ax is None:
            plt.close('all')
            fig, ax = plt.subplots()

        ax.set_xlim(self.x0, self.x1)
        ax.set_ylim(self.y0, self.y1)
        ax.set_xticks(np.linspace(self.x0, self.x1, 5))
        ax.set_yticks(np.linspace(self.y0, self.y1, 5))
        ax.set_aspect(1)
        
        vertices = self.vertices
        vertex_num = self.vertices.shape[0]

        for i in range(vertex_num):
            vertex_0 = vertices[i]
            vertex_1 = vertices[(i+1)%vertex_num]
            ax.add_line(Line2D([vertex_0[0], vertex_1[0]], [vertex_0[1], vertex_1[1]], linewidth=line_width, color=color))

        if save:
            ensure_path(save_path)
            plt.savefig(save_path + save_name)

        return ax
    
    def plot_random_xy_cv(self, img=None, save=True, save_path='./', save_name='arena_random_xy.png', num=100, color=(0,255,0), plot_arena=True, **kw):
        if img is None:
            res = search_dict(kw, ['res, resolution'], default=100)
            res_x, res_y = get_res_xy(res, self.width, self.height)
            if plot_arena:
                img = self.plot_arena(save=False, res=res)
            else:
                img = np.zeros([res_x, res_y, 3], dtype=np.uint8)
                img[:,:,:] = (255, 255, 255)
        else:
            res_x, res_y = img.shape[0], img.shape[1]
        points = self.get_random_xy(num=100)
        points_int = get_int_coords_np(points, self.xy_range, res_x, res_y)
        for point in points_int:
            cv.circle(img, (point[0], point[1]), radius=0, color=color, thickness=4)
        if save:
            ensure_path(save_path)
            cv.imwrite(save_path + save_name, img[:, ::-1, :])
        return img
    def plot_random_xy_plt(self, ax=None, save=True, save_path='./', save_name='arena_random_xy.png', num=100, color=(0.0,1.0,0.0), plot_arena=True, **kw):
        if ax is None:
            figure, ax = plt.subplots()
            if plot_arena:
                self.plot_arena_plt(ax, save=False)
            else:
                ax.set_xlim(self.x0, self.x1)
                ax.set_ylim(self.y0, self.y1)
                ax.set_aspect(1) # so that x and y has same unit length in image.
        points = self.get_random_xy(num=100)
        ax.scatter(points[:,0], points[:,1], marker='o', color=color, edgecolors=(0.0,0.0,0.0), label='Start Positions')
        plt.legend()

        if save:
            ensure_path(save_path)
            plt.savefig(save_path + save_name)
        return ax

    def write_info(self, save_path='./', save_name='Arena Dict.txt'):
        ensure_path(save_path)
        utils.write_dict()
        with open(save_path + save_name, 'w') as f:
            return # to be implemented

class Arena_Circle(Arena):
    def __init__(self, dict_, options=None, load=False):
        super().__init__(dict_, options, load)

        self.type = self.dict['type'] = 'circle'
        self.radius = get_from_dict(self.dict, 'radius', default=None, write_default=True)
        self.center_coord = get_from_dict(self.dict, 'center_coord', default=[0.0, 0.0], write_default=True)
        if isinstance(self.center_coord, list):
            self.center_coord = np.array(self.center_coord, dtype=np.float)
        
        # By opencv convention, origin is at the top-left corner of the pircture.
        self.x0 = self.center_coord[0] - self.radius
        self.x1 = self.center_coord[0] + self.radius
        self.y0 = self.center_coord[1] - self.radius
        self.y1 = self.center_coord[1] + self.radius
        self.xy_range = (self.x0, self.x0, self.x1, self.y1)
        self.square_min_size = self.width = self.height = 2 * self.radius
        self.border_region_width = search_dict(self.dict, ['border_region_width'], default=0.03 * self.square_min_size, write_default=True)

        self.get_random_max = self.get_random_square = self.get_random_max_rectangle
        
        self.avoid_border = self.avoid_border_circle
        self.out_of_region = self.out_of_region_circle
        self.get_random_xy = self.get_random_xy_circle
        
        self.plot_arena = self.plot_arena_plt
        #self.plot_arena(save_path='./anal/', save_name='arena_plot_circle.png')
        #self.print_info()
    def print_info(self):
        print('Arena_Circle: ', end='')
        print('center_coord:(%.1f, %.1f)'%(self.center_coord[0], self.center_coord[1]))
        print('vertices:', end='')
        for index in range(self.edge_num):
            print('(%.1f, %.1f)'%(self.vertices[index][0], self.vertices[index][1]), end='')
        print('\n')
    def get_nearest_border(self, points, ):
        return
    def get_random_xy_circle(self, num=100, thres=None):    
        if thres is None or thres in ['default']:
            thres = self.border_region_width
        count = 0
        points_list = []
        while(count < num):
            points = self.get_random_xy_max(int(2.2*num))            
            points = np.delete(points, self.out_of_region(points, thres=thres), axis=0)
            points_list.append(points)
            count += points.shape[0]
            #print('total valid points:%d'%count)
        points = np.concatenate(points_list, axis=0)
        if count > num:
            #print(points.shape)
            points = np.delete(points, random.sample(range(count), count - num), axis=0)
            #print(points.shape)
        return points
    def get_dist_from_center(self, points):
        vec_from_center = points - self.center_coord[np.newaxis, :] # [point_num, (x, y)]
        dist_from_center = np.linalg.norm(vec_from_center, ord=2, axis=-1)[:, np.newaxis] # [point_num]
        return dist_from_center
    def get_dist_and_theta_from_center(self, xy): # xy: [point_num, 2]
        vec_from_center = xy - self.center_coord[np.newaxis, :] # [point_num, (x, y)]
        dist_from_center = np.linalg.norm(vec_from_center, ord=2, axis=-1) # [point_num]
        theta_from_center = np.arctan2(xy[:, 1], xy[:, 0])
        return dist_from_center, theta_from_center

    def out_of_region_circle(self, points, thres=None): # coords:[point_num, (x, y)]. This method is only valid for convex polygon.
        if thres is None or thres in ['default']:
            thres = self.border_region_width
        dist_from_center = self.get_dist_from_center(points).squeeze()
        #print(dist_from_center)
        return np.argwhere( dist_from_center > (self.radius - thres) ).squeeze() # indices of out-of-region points: [out_of_region_point_num]
    def avoid_border_circle(self, xy, theta, thres=None): # xy: [batch_size, (x, y)], theta: velocity direction.
        if thres is None or thres in ['default']:
            thres = self.border_region_width
        dist_from_center, theta_from_center = self.get_dist_and_theta_from_center(xy) # [batch_size]

        #print(theta_from_center)
        theta_offset = theta - theta_from_center # [batch_size] 
        theta_offset = np.mod(theta_offset + np.pi, 2*np.pi) - np.pi # range:(-pi, pi)
        #print(theta_offset.shape)
       
        towards_wall = np.abs(theta_offset) < np.pi/2
        is_near_wall = (dist_from_center > (self.radius - thres)) * towards_wall # [batch_size]
        #print(is_near_wall.shape)
        #print(dist_from_center.shape)

        theta_adjust = np.zeros_like(theta) # zero arrays with the same shape as theta.
        theta_adjust[is_near_wall] = np.sign(theta_offset[is_near_wall]) * ( np.pi/2 - np.abs(theta_offset[is_near_wall]) ) # turn to direction parallel to wall

        return is_near_wall, theta_adjust

    def avoid_border_free(self, position, theta, arena_dict=None):
        is_near_wall = ( np.zeros_like(theta) > 1.0 )
        theta_adjust = np.zeros_like(theta)
        return is_near_wall, theta_adjust

    def avoid_border_square(self, position, hd, box_width=None, box_height=None, arena_dict=None, thres=0.0):
        x = position[:,0]
        y = position[:,1]
        dists = [self.x1 - x, self.y1 - y, x - self.x0, y - self.y0] #distance to all edges (right, top, left, down)
        d_wall = np.min(dists, axis=0) #distance to nearst wall
        angles = np.arange(4)*np.pi/2 
        theta = angles[np.argmin(dists, axis=0)]
        hd = np.mod(hd, 2*np.pi)
        a_wall = hd - theta
        a_wall = np.mod(a_wall + np.pi, 2*np.pi) - np.pi
        
        is_near_wall = (d_wall < thres) * (np.abs(a_wall) < np.pi/2)
        theta_adjust = np.zeros_like(hd) #zero arrays with the same shape of hd.
        theta_adjust[is_near_wall] = np.sign(a_wall[is_near_wall])*(np.pi/2 - np.abs(a_wall[is_near_wall]))
        return is_near_wall, theta_adjust
    def plot_border(self, img): # img: [H, W, C], np.uint8. 
        return # to be implemented   
    def plot_arena_plt(self, ax=None, save=True, save_path='./', save_name='arena_random_xy.png', line_width=2, color=(0,0,0)):
        if ax is None:
            figure, ax = plt.subplots()

        ax.set_xlim(self.x0, self.x1)
        ax.set_ylim(self.y0, self.y1)
        ax.set_xticks(np.linspace(self.x0, self.x1, 5))
        ax.set_yticks(np.linspace(self.y0, self.y1, 5))
        ax.set_aspect(1)

        circle = plt.Circle(self.center_coord, self.radius, color=(0.0, 0.0, 0.0), fill=False)        
        ax.add_patch(circle)
        if save:
            ensure_path(save_path)
            plt.savefig(save_path + save_name)

        return ax
    
    def plot_random_xy_plt(self, ax=None, save=True, save_path='./', save_name='arena_random_xy.png', num=100, color=(0.0,1.0,0.0), plot_arena=True, **kw):
        if ax is None:
            figure, ax = plt.subplots()
            if plot_arena:
                self.plot_arena_plt(ax, save=False)
            else:
                ax.set_xlim(self.x0, self.x1)
                ax.set_ylim(self.y0, self.y1)
                ax.set_aspect(1) # so that x and y has same unit length in image.
        points = self.get_random_xy(num=100)
        ax.scatter(points[:,0], points[:,1], marker='o', color=color, edgecolors=(0.0,0.0,0.0), label='Start Positions')
        plt.legend()

        if save:
            ensure_path(save_path)
            plt.savefig(save_path + save_name)
        return ax

    def write_info(self, save_path='./', save_name='Arena Dict.txt'):
        ensure_path(save_path)
        utils.write_dict()
        with open(save_path + save_name, 'w') as f:
            return # to be implemented

