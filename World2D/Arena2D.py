import numpy as np
import random
import torch

import matplotlib as mpl
from matplotlib import pyplot as plt

import utils
import World2D
import utils_torch
from utils_torch.attrs import *
class Arena2D():
    def __init__(self, param=None):
        self.param = param
        if param is not None:
            self.param = param
    def InitFromParam(self, param=None):
        if param is not None:
            self.param = param
        else:
            param = self.param
        EnsureAttrs(param, "Init", default=[])
        EnsureAttrs(param, "Shapes", default=[])
        if len(param.Shapes)==0:
            raise Exception()
        self.Shapes = []
        for ShapeParam in param.Shapes:
            Shape = World2D.Shapes2D.BuildShape(ShapeParam)
            Shape.InitFromParam()
            self.Shapes.append(Shape)
        self.CalculateBoundaryBox()

        
    def CalculateBoundaryBox(self):
        param = self.param
        BoundaryBoxes = []
        for Shape in self.Shapes:
            BoundaryBoxes.append(GetAttrs(Shape.param.BoundaryBox))
        BoundaryBoxes = utils_torch.ToNpArray(BoundaryBoxes) # [ShapeNum, (XMin, YMin, XMax, YMax)]
        if len(BoundaryBoxes.shape)==1:
            BoundaryBoxes = BoundaryBoxes[np.newaxis, :]
        
        BoundaryBox = [np.min(BoundaryBoxes[:, 0]), np.min(BoundaryBoxes[:, 1]),
            np.max(BoundaryBoxes[:, 2]), np.max(BoundaryBoxes[:, 3])]
        
        if HasAttrs(param, "BoundaryBox") and isinstance(GetAttrs(param.BoundaryBox), list):
            if not utils_torch.geometry2D.RectangleAContainsRectangleB(GetAttrs(param.BoundaryBox), BoundaryBox):
                raise Exception()
            else:
                return
        else:
            SetAttrs(param, "BoundaryBox", BoundaryBox)

        SetAttrs(param, "BoundaryBox.XMin", GetAttrs(param.BoundaryBox)[0])
        SetAttrs(param, "BoundaryBox.YMin", GetAttrs(param.BoundaryBox)[1])
        SetAttrs(param, "BoundaryBox.XMax", GetAttrs(param.BoundaryBox)[2])
        SetAttrs(param, "BoundaryBox.YMax", GetAttrs(param.BoundaryBox)[3])
        
        BoundaryBox = param.BoundaryBox
        SetAttrs(param, "BoundaryBox.Width", BoundaryBox.XMax - BoundaryBox.XMin)
        SetAttrs(param, "BoundaryBox.Height", BoundaryBox.YMax - BoundaryBox.YMin)
        SetAttrs(param, "BoundaryBox.Size", max(BoundaryBox.Width, BoundaryBox.Height))
    def PlotArena(self, ax=None, Save=True, SavePath="./Arena2D-Plot.png"):
        param = self.param
        if ax is None:
            plt.close("all")
            fig, ax = plt.subplots()
        for Shape in self.Shapes:
            Shape.PlotShape(ax, Save=False, SetXYRange=False)
        utils_torch.plot.SetHeightWidthRatio(ax, 1.0)
        utils_torch.plot.SetAxRangeFromBoundaryBox(ax, param.BoundaryBox)
        if Save:
            utils_torch.EnsureFileDir(SavePath)
            plt.savefig(SavePath)
        return ax
    def PlotRandomInternalPoints(self, PointNum, SavePath):
        fig, ax = plt.subplots()
        Points = self.GenerateRandomInternalXYs(PointNum)
        self.PlotBoundary(ax)
        utils_torch.plot.PlotPointsNp(Points)
        plt.savefig(SavePath)
    def IsInside(self, Points, MinDistance2Border=0.0):
        # @param Points: numpy.ndarray with shape [PointNum, (x, y)]
        # Requires Arena to be Simply Connected.
        PointNum = Points.shape[0]
        isInside = np.ones((PointNum), dtype=np.bool8)
        for Shape in self.Shapes:
            isInside = isInside * Shape.IsInside(Points, MinDistance2Border=MinDistance2Border)
        return isInside
    def IsOutside(self, Points, MinDistance2Border=0.0):
        # @param Points: numpy.ndarray with shape [PointNum, (x, y)]
        # @param MinDistance2Border: to be implemented
        return ~self.IsInside(Points, MinDistance2Border=MinDistance2Border)
    def CheckCollision(self, XY, dXY):
        # Planned Trajectory: Points -> Points + Vectors
        # @return: np.ndarray with shape [PointNum]. 
        PointNum = XY.shape[0]
        ShapeNum = len(self.Shapes)
        LambdasOfAllShapes = np.ones((PointNum, ShapeNum), dtype=np.float32)
        Norms = np.zeros((PointNum, ShapeNum, 2), dtype=np.float32)
        for ShapeIndex, Shape in enumerate(self.Shapes):
            Collision = Shape.CheckCollision(XY, dXY)
            if Collision.Num > 0:
                LambdasOfAllShapes[Collision.Indices, ShapeIndex] = Collision.Lambdas
                Norms[Collision.Indices, ShapeIndex, :] = Collision.Norms
        #LambdasOfAllShapes = np.stack(Lambdas, axis=1) # [List: ShapeNum][np: PointNum] --> [np: PointNum, ShapeNum]
        Lambdas = np.min(LambdasOfAllShapes, axis=1) # [PointNum, ShapeNum] --> [PointNum]
        CollisionPointIndices = np.argwhere(Lambdas<1.0).squeeze() # [CollisionPointNum]
        LambdasOfCollisionShapes = LambdasOfAllShapes[CollisionPointIndices, :]
        if len(LambdasOfCollisionShapes.shape)==1:
            LambdasOfCollisionShapes = LambdasOfCollisionShapes[np.newaxis, :]
        CollisionShapeIndices = np.argmin(LambdasOfCollisionShapes, axis=1)

        self.ReportCollision(XY[CollisionPointIndices], dXY[CollisionPointIndices])

        return utils_torch.json.JsonObj2PyObj({
            "Indices": CollisionPointIndices,
            "Lambdas": np.atleast_1d(Lambdas[CollisionPointIndices]), # [CollisionPointNum]
            "Norms": Norms[CollisionPointIndices, CollisionShapeIndices, :] # [CollisionPointNum, 2]
        })
    def ReportCollision(XY, dXY, XYCollision=None):
        return
    def PlotInsideMask(self, ax=None, Save=False, SavePath=utils.ArgsGlobal.SaveDir + "Arena2D-InsideMask.png"):
        if ax is None:
            fig, ax = plt.subplots()
        mask = self.GetInsideMask(ResolutionX=50, ResolutionY=50)
        mask = mask.astype(np.int32)
        utils_torch.plot.PlotMatrix(ax, mask)
        if Save:
            plt.savefig(SavePath, format="png")
    def GetInsideMask(self, BoundaryBox=None, ResolutionX=None, ResolutionY=None):
        param = self.param
        if BoundaryBox is None:
            BoundaryBox = param.BoundaryBox
        mask = np.zeros((ResolutionX, ResolutionY), dtype=np.bool8)
        XYs = utils_torch.geometry2D.LatticeXYs(BoundaryBox, ResolutionX, ResolutionY, Flatten=True) # [ResolutionX * ResolutionY, (x, y)]
        isInside = self.IsInside(XYs)
        isInsideIndex = np.argwhere(isInside)[:, 0]
        mask[isInsideIndex // ResolutionY, isInsideIndex % ResolutionY] = 1
        return mask
    #@abc.abstractmethod
    def Getrandom_xy(self): # must be implemented by child class.
        return
    def GenerateRandomPointsInBoundaryBox(self, Num):
        param = self.param
        Points = np.stack([np.random.uniform(param.BoundaryBox.XMin, param.BoundaryBox.XMax, Num), 
            np.random.uniform(param.BoundaryBox.YMin, param.BoundaryBox.YMax, Num)], axis=1) #[points_num, (x, y)]
        return Points
    def GenerateRandomInternalXYs(self, Num=100, MinDistance2Border=0.0):
        PointNum = 0
        PointList = []
        while(PointNum < Num):
            Points = self.GenerateRandomPointsInBoundaryBox(int(2.2 * Num))
            PointsOutsideIndices = np.argwhere(self.IsOutside(Points, MinDistance2Border=MinDistance2Border))       
            Points = np.delete(Points, PointsOutsideIndices, axis=0)
            PointList.append(Points)
            PointNum += Points.shape[0]
            #print('total valid points:%d'%count)
        Points = np.concatenate(PointList, axis=0)
        if PointNum > Num:
            Points = np.delete(Points, random.sample(range(PointNum), PointNum - Num), axis=0)
        return Points
    def Getrandom_xy_max(self, num=100):
        xs = np.random.uniform(self.x0, self.x1, num) # x_0
        ys = np.random.uniform(self.y0, self.y1, num) # y_0
        xys = np.stack([xs,ys], axis=1) # np.stack will insert a dimension at designated axis. [num, 2]
        #print('Getrandom_xy_max: points shape:'+str(xys.shape))
        return xys
    def save(self, dir_):
        with open(dir_, 'wb') as f:
            net = self.to(torch.device('cpu'))
            torch.save(net.dict, f)
            net = self.to(self.device)
    def Getmask(self, res=None, res_x=None, res_y=None, points_grid=None):
        #print('res_x: %d res_y: %d'%(res_x, res_y))
        if res is not None:
            res_x, res_y = Getres_xy(res, self.width, self.height)
        if points_grid is not None:
            if points_grid.shape[0] != res_x * res_y:
                raise Exception('Arena.Getmask: points_grid shape must be consistent with res_x, res_y.')
        else:
            points_grid_int = np.empty(shape=[res_x, res_y, 2], dtype=np.int) # coord
            for i in range(res_x):
                points_grid_int[i, :, 1] = i # y_index
            for i in range(res_y):
                points_grid_int[:, i, 0] = i # x_index
            points_grid = Getfloat_coords_np( points_grid_int.reshape(res_x * res_y, 2), self.xy_range, res_x, res_y ) # [res_x * res_y, 2]
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

__MainClass__ = Arena2D