import utils_torch
from utils_torch import get_from_dict

import Environments

class ArenaPolygon(Environments.Arena):
    def __init__(self, param=None):
        super().__init__(dict_, load)
        if param is not None:
            self.InitFromParams(param)
    def InitFromParams(self, param):

        #set_instance_variable(self, self.dict, ['width', 'height', 'type'])
        self.width = self.dict['width'] # maximum rectangle
        self.height =self.dict['height']
        self.type_ = self.type = self.dict['type']
        
        self.center_coord = get_from_dict(self.dict, 'center_coord', default=[0.0, 0.0], write_default=True)
        #print(self.center_coord)
        #print(self.width)
        #print(self.height)
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
