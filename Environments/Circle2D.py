
import numpy as np
import utils_torch
from utils_torch.attrs import EnsureAttrs, SetAttrs, HasAttrs, MatchAttrs

import Environments

class Circle2D(Environments.Arena2D):
    def __init__(self, param=None):
        super().__init__()
        if param is not None:
            self.InitFromParams(param)

    def InitFromParams(self, param):
        self.param = param

        EnsureAttrs()

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