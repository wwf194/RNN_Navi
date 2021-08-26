import math
import numpy as np
import random

import Environments

import utils
from utils.arena import CalculateDistanceToEdges, Vertices2VertexPairs

import utils_torch
from utils_torch import get_from_dict
from utils_torch.attrs import CheckAttrs, EnsureAttrs, GetAttrs, HasAttrs, SetAttrs
from utils_torch.math import xy2polar

import matplotlib as mpl
from matplotlib import pyplot as plt

class Polygon2D(Environments.Arena2D):
    def __init__(self, param=None):
        super().__init__(param)
        if param is not None:
            self.InitFromParam(param)
    def InitFromParam(self, param):
        # Check Arena Type
        if GetAttrs(param, "Type") in ["Polygon", "Rectangle", "Triangle"]:
            SetAttrs(param.Type, "Polygon")
        else:
            raise Exception
        EnsureAttrs(param, "Subtype", default="RegularPolygon")
        EnsureAttrs(param, "Initialize.Method", default="FromVertices")
        if param.Initialize.Method in ["FromVertices", "FromVertex"]:
            self.CalculateEdgesFromPoints()
        elif param.Initialize.Method in ["CenterRadiusTheta"]:
            CheckAttrs(param.Subtype, value="RegluarPolygon")
            self.InitRegularPolygon()
            self.CalculateEdgesFromPoints()
        else:
            raise Exception()

        self.CalculateEdgeVectorsFromEdges()
        self.CalculateEdgeNorms()
        
        
        self.CalculateBoundaryBox()
        SetAttrs(param.Edges, "Num", len(param.Edges))
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
    def GenerateRandomInternalPositions(self):
        return
    def CalculateRegularPolygonVertices(self):
        param = self.param
        if not HasAttrs(param, "Edges.Num"):
            if not HasAttrs(param, "Vertices.Num"):
                raise Exception()
            SetAttrs(param, "Edges.Num", param.Vertices.Num)
        edge_num = search_dict(kw, ["point_num", "edge_num"])
        square_max_size = search_dict(kw, ["box_width", "edge_length", "box_size", "square_max_size", "square_size"])
        direct_offset = search_dict(kw, ["direct", "direct_offset"])
        center_coord = search_dict(kw, ["center", "center_coord"], default=[0.0, 0.0])
        Radius = square_max_size / 2
        DirectionIncrement = math.pi * 2 / edge_num
        PointsNp = np.zeros([param.Edges.Num, 2])
        
        Vertices = []
        if direct_offset is None:
            if RandomDirect:
                theta_now = random.random() * DirectionIncrement # srandom.random() generate a random float in (0, 1)
            else:
                theta_now = 0.0
        else:
            Direction = direct_offset
        for num in range(param.Edges.Num):
            x, y = utils_torch.math.polar2xy(Radius, Direction)
            Vertices.append([x + param.Initialize.Center[0], y + param.Initialize.Center[1]])
            Direction += DirectionIncrement
        return PointsNp.tolist()
    def isOutSide(self, Points):
        return ~self.isInside(Points)
    def IsOutside(self, Points, ThresholdDistance=None): # coords:[point_num, (x, y)]. This method is only valid for convex polygon.
        if ThresholdDistance is None:
            ThresholdDistance = self.OutsideThresholdDistance
        Distance = self.Distance2Edges(Points) # [PointNum, EdgeNum]
        return np.sum(Distance < ThresholdDistance, axis=1) > 0
    def CalculatEdgesFromVertices(self):
        param = self.param
        EnsureAttrs(param.Vertices, "Num", len(GetAttrs(param.Vertices)))
        SetAttrs(param, "Edges", Vertices2VertexPairs(GetAttrs(param.Vertices)))
        SetAttrs(param.Edges, "Num")
        VerticesNp = np.array(param.Points, dtype=np.float32)
        VerticesNp = np.concatenate((VerticesNp, VerticesNp[0][np.newaxis, :]), axis=0)
    def CalculateEdgesInfo(self):
        param = self.param
        SetAttrs(param.Edges, "Vector", utils.arena.VertiexPairs2Vectors(GetAttrs(param.Edges)))
        SetAttrs(param.Edges, "Norm", utils.arena.Vectors2Norms(GetAttrs(param.Edges.Vector)))
        SetAttrs(param.Edges, "Norm", "Direction", utils.arena.Vectors2Directions(GetAttrs(param.Edges.Norm)))
        self.Edges = np.array(param.Edges, dtpe=np.float32)
        self.Edges.Vector = np.array(param.Edges.Vector, dtype=np.float32)
        self.Edges.Norm = np.array(param.Edges.Norm, dtype=np.float32)
        self.Edges.Norm.Direction = np.array(param.Edges.Norm.Direction, dtype=np.float32)
    def CalculateBoundaryBox(self):
        param = self.param
        # Calculate Boundary Box
        if not HasAttrs(param, "BoundaryBox"):
            EdgesNp = np.array(param.Edges, dtype=np.float32) # [VertexNum, (x, y)]
            xMin = np.min(EdgesNp[:, 0])
            xMax = np.max(EdgesNp[:, 0])
            yMin = np.min(EdgesNp[:, 1])
            yMax = np.max(EdgesNp[:, 1])
            SetAttrs(param, "BoundaryBox", value=[[xMin, yMin], [xMax, yMax]])
            SetAttrs(param.BoundaryBox, "xMin", xMin)
            SetAttrs(param.BoundaryBox, "xMax", xMax)
            SetAttrs(param.BoundaryBox, "yMin", yMin)
            SetAttrs(param.BoundaryBox, "yMax", yMax)
    def PrintInfo(self):
        utils.add_log('Arena_Polygon: edge_num:%d'%(self.edge_num))
        print('center_coord:(%.1f, %.1f)'%(self.center_coord[0], self.center_coord[1]))
        print('vertices:', end='')
        for index in range(self.edge_num):
            print('(%.1f, %.1f)'%(self.vertices[index][0], self.vertices[index][1]), end='')
        print('\n')
    
    def Vector2NearstBorder(self, Points):
        return

    def Distance2Edges(self, xy):
        return utils.arena.Distance2Edges(xy, self.VerticesNp, self.EdgeNormsNp)
    def avoid_border_polygon(self, Points, Directions, ThresholdDistance=None): # theta: velocity direction.
        Info = self.Info
        if ThresholdDistance is None:
            ThresholdDistance = self.OutsideThresholdDistance
        Distance2Edges = self.Distance2Edges(Points) # [PointNum, EdgeNum]
        
        Vector2NearstEdge = self.Vector2NearstEdge(Points)

        Distance2NearstEdge, Direction2NearstEdge = utils_torch.math.Vectors2NormsDirectionsNp(Vector2NearstEdge, axis=1)

        isTowardsWall = utils_torch.math.isAcuteAngle(Direction2NearstEdge, Directions)

        isNearWall = 

        IsNearAndTowardsWll

        TowardsNearstEdge = 
        
        DistanceMin = np.min(Distance2Edges, axis=1) # [PointNum]. Smallest Distance.
        NearstEdgeIndex = np.argmin(Distance2Edges, axis=1) # [PointNum]
        NearstEdgeNormDirection = Info.Edges.Norm.Direction()
        np.argmin(Distance, axis=1)

        
        
        

        norms_theta = self.edge_norms_theta # [edge_num], range:(-pi, pi)
        nearst_theta = norms_theta[np.argmin(dists, axis=1)] # (batch_size). Here index is tuple.
        
        theta = np.mod(theta, 2*np.pi) # range: (0, 2*pi)
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
    def PlotArenaCv(self, img=None, line_color=(0,0,0), line_width=1, line_type=4, save=False, save_path='./', save_name='arena_plot.png', **kw):# line_color: (b, g, r)
        if img is None:
            res = search_dict(kw, ['res, resolution'], default=100)
            res_x, res_y = get_res_xy(res, self.width, self.height)
            img = np.zeros([res_x, res_y, 3], dtype=np.uint8)
            img[:,:,:] = (255, 255, 255)

        vertices, width, height = self.dict['vertices'], self.width, self.height
        vertex_num = vertices.shape[0]     
        #print('xy_range:%s'%str(xy_range))
        for i in range(vertex_num):
            vertex_0 = utils_torch.plot.Float2PixelIndex(vertices[i, 0], vertices[i, 1], self.xy_range, res_x, res_y)
            vertex_1 = utils_torch.plot.Float2PixelIndex(vertices[(i+1)%vertex_num, 0], vertices[(i+1)%vertex_num, 1], self.xy_range, res_x, res_y)
            #print('plot line: (%d, %d) to (%d, %d)'%(vertex_0[0], vertex_0[1], vertex_1[0], vertex_1[1]))
            cv.line(img, vertex_0, vertex_1, line_color, line_width, line_type) # line_width seems to be in unit of pixel.
        
        if save:
            EnsurePath(save_path)
            #if save_name is None:
            #    save_name = 'arena_plot.png'
            cv.imwrite(save_path + save_name, img[:, ::-1, :]) # so that origin is in left-bottom corner.
        return img
    def PlotArena(self, ax=None, Save=True, SavePath="./", line_width=2, color=(0,0,0)):
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
        if Save:
            EnsurePathOfFile(SavePath)
            plt.savefig(SavePath)
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
            EnsurePath(save_path)
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
            EnsurePath(save_path)
            plt.savefig(save_path + save_name)
        return ax
