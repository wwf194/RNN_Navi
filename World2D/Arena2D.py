import numpy as np
import random
import torch

import matplotlib as mpl
from matplotlib import pyplot as plt

import World2D
import utils_torch
from utils_torch.attrs import *

class Arena2D():
    def __init__(self, param=None, data=None, **kw):
        utils_torch.model.InitForModel(self, param, data, ClassPath="World2D.Arena2D", **kw)
    def InitFromParam(self, param=None, IsLoad=False):
        if param is not None:
            self.param = param
        else:
            param = self.param
            cache = self.cache
        cache.IsLoad = IsLoad
        cache.IsInit = not IsLoad
        cache.Modules = utils_torch.EmptyPyObj()
        if cache.IsInit:
            EnsureAttrs(param, "Init", default=[])
            EnsureAttrs(param, "Shapes", default=[])
        
        if len(param.Shapes)==0:
            raise Exception()
        self.Shapes = []
        for Index, ShapeParam in enumerate(GetAttrs(param.Shapes)):
            Shape = World2D.Shapes2D.BuildShape(ShapeParam, LoadDir=cache.LoadDir)
            Shape.InitFromParam(IsLoad=cache.IsLoad)
            self.Shapes.append(Shape)
            setattr(cache.Modules, "Shape%d"%Index, Shape)
        if cache.IsInit:
            self.CalculateBoundaryBox()
    def GetBoundaryBox(self):
        return self.param.BoundaryBox
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
    def PlotArena(self, ax=None, Save=False, SavePath=None):
        param = self.param
        if ax is None:
            plt.close("all")
            fig, ax = plt.subplots()
        for Shape in self.Shapes:
            Shape.PlotShape(ax, Save=False, SetXYRange=False)
        utils_torch.plot.SetHeightWidthRatio(ax, 1.0)
        utils_torch.plot.SetAxRangeAndTicksFromBoundaryBox(ax, param.BoundaryBox)        
        utils_torch.plot.SaveFigForPlt(Save, SavePath)
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

        return utils_torch.PyObj({
            "Indices": CollisionPointIndices,
            "Lambdas": np.atleast_1d(Lambdas[CollisionPointIndices]), # [CollisionPointNum]
            "Norms": Norms[CollisionPointIndices, CollisionShapeIndices, :] # [CollisionPointNum, 2]
        })
    def ReportCollision(XY, dXY, XYCollision=None):
        return
    def PlotInsideMask(self, ax=None, Save=False, SavePath=None):
        if SavePath is None:
            SavePath = utils_torch.GetMainSaveDir() + "Arenas/" + "Arena2D-InsideMask.png"
        if ax is None:
            fig, ax = plt.subplots()
        mask = self.GetInsideMask(ResolutionX=50, ResolutionY=50)
        mask = mask.astype(np.int32)
        utils_torch.plot.PlotMatrix(ax, mask)
        if Save:
            utils_torch.EnsureFileDir(SavePath)
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
    def GenerateRandomXYsInBoundaryBox(self, Num):
        param = self.param
        Points = np.stack([np.random.uniform(param.BoundaryBox.XMin, param.BoundaryBox.XMax, Num), 
            np.random.uniform(param.BoundaryBox.YMin, param.BoundaryBox.YMax, Num)], axis=1) #[points_num, (x, y)]
        return Points
    def GenerateRandomInternalXYs(self, Num=100, MinDistance2Border=0.0):
        PointNum = 0
        PointList = []
        while(PointNum < Num):
            Points = self.GenerateRandomXYsInBoundaryBox(int(2.2 * Num))
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

__MainClass__ = Arena2D
utils_torch.model.SetMethodForWorldClass(__MainClass__)