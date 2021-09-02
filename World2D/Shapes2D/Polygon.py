

import math
import numpy as np
import random

import utils
from utils_torch.geometry import Vertices2VertexPairs

import utils_torch
from utils_torch.attrs import CheckAttrs, EnsureAttrs, GetAttrs, HasAttrs, SetAttrs
from utils_torch.geometry import XY2Polar

import matplotlib as mpl
from matplotlib import pyplot as plt

from World2D.Shapes2D import Shape2D

class Polygon(Shape2D):
    def __init__(self, param=None):
        if param is not None:
            self.InitFromParam(param)
    def InitFromParam(self, param):
        self.param = param
        # Check Shape Type
        if GetAttrs(param, "Type") in ["Polygon", "Rectangle", "Triangle"]:
            SetAttrs(param.Type, "Polygon")
        else:
            raise Exception()
        EnsureAttrs(param, "Subtype", default="RegularPolygon")
        EnsureAttrs(param, "Initialize.Method", default="FromVertices")
        if param.Initialize.Method in ["FromVertices", "FromVertex"]:
            self.CalculatEdgesFromVertices()
        elif param.Initialize.Method in ["CenterRadiusTheta"]:
            CheckAttrs(param.Subtype, value="RegularPolygon")
            self.InitRegularPolygon()
            self.CalculatEdgesFromVertices()
        else:
            raise Exception()
        
        self.CalculateEdgesInfo()
        self.CalculateBoundaryBox()

    def GenerateRandomInternalPositions(self):
        return
    def InitRegularPolygon(self):
        param = self.param
        if not HasAttrs(param, "Edges.Num"):
            if not HasAttrs(param, "Vertices.Num"):
                raise Exception()
            SetAttrs(param, "Edges.Num", param.Vertices.Num)        

        EnsureAttrs(param.Initialize, "")

        DirectionIncrement = math.pi * 2 / param.Edges.Num
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
            x, y = utils_torch.math.Polar2XY(Radius, Direction)
            Vertices.append([x + param.Initialize.Center[0], y + param.Initialize.Center[1]])
            Direction += DirectionIncrement
        return PointsNp.tolist()
    def isOutSide_(self, Points):
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
        VerticesNp = np.array(GetAttrs(param.Points), dtype=np.float32)
        VerticesNp = np.concatenate((VerticesNp, VerticesNp[0][np.newaxis, :]), axis=0)
    def CalculateEdgesInfo(self):
        param = self.param
        SetAttrs(param.Edges, "Num", len(param.Edges))
        SetAttrs(param.Edges, "Vector", utils.arena.VertexPairs2Vectors(GetAttrs(param.Edges)))
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
            VerticesNp = np.array(param.Vertices, dtype=np.float32) # [VertexNum, (x, y)]
            xMin = np.min(VerticesNp[:, 0])
            xMax = np.max(VerticesNp[:, 0])
            yMin = np.min(VerticesNp[:, 1])
            yMax = np.max(VerticesNp[:, 1])
            SetAttrs(param, "BoundaryBox", value=[xMin, yMin, xMax, yMax])
            SetAttrs(param.BoundaryBox, "xMin", xMin)
            SetAttrs(param.BoundaryBox, "xMax", xMax)
            SetAttrs(param.BoundaryBox, "yMin", yMin)
            SetAttrs(param.BoundaryBox, "yMax", yMax)
    def PrintInfo(self):
        utils.AddLog('Arena_Polygon: edge_num:%d'%(self.edge_num))
        print('center_coord:(%.1f, %.1f)'%(self.center_coord[0], self.center_coord[1]))
        print('vertices:', end='')
        for index in range(self.edge_num):
            print('(%.1f, %.1f)'%(self.vertices[index][0], self.vertices[index][1]), end='')
        print('\n')
    def Vector2NearstBorder(self, Points):
        return
    def Distance2Edges(self, xy):
        return utils.arena.Distance2Edges(xy, self.VerticesNp, self.EdgeNormsNp)
    def isMovingOutside(self, Points, NextPoints):
        return self.isInside(Points) * self.isOutside(NextPoints)
    def CheckCollision(self, Points, Steps, PointsNext):
        PointNum = Points.shape[0]
        Lambdas = []
        for Index in range(self.Edges.Num):
            Lambdas.append(utils_torch.geometry.InterceptRatio(self.EdgesNp[np.newaxis, Index, 0], self.EdgesNp[np.newaxis, Index, 1], Points, PointsNext))
        Lambdas = np.stack(Lambdas, axis=1) # [List: EdgeNum][np: PointNum] --> [np: PointNum, ShapeNum]
        return np.min(Lambdas, axis=1) # [PointNum, ShapeNum] --> [PointNum]
    def PlotEdgesPlt(self, ax): # img: [H, W, C], np.uint8.
        utils_torch.plot.PlotLines(ax, self.EdgesNp)
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
            res_x, res_y = Getres_xy(res, self.width, self.height)
            if plot_arena:
                img = self.plot_arena(save=False, res=res)
            else:
                img = np.zeros([res_x, res_y, 3], dtype=np.uint8)
                img[:,:,:] = (255, 255, 255)
        else:
            res_x, res_y = img.shape[0], img.shape[1]
        points = self.Getrandom_xy(num=100)
        points_int = Getint_coords_np(points, self.xy_range, res_x, res_y)
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
        points = self.Getrandom_xy(num=100)
        ax.scatter(points[:,0], points[:,1], marker='o', color=color, edgecolors=(0.0,0.0,0.0), label='Start Positions')
        plt.legend()
        if save:
            EnsurePath(save_path)
            plt.savefig(save_path + save_name)
        return ax