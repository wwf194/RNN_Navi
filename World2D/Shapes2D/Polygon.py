

import math
import numpy as np
import random

import utils
from utils_torch.geometry2D import Vertices2VertexPairs

import utils_torch
from utils_torch.attrs import CheckAttrs, EnsureAttrs, GetAttrs, HasAttrs, SetAttrs
from utils_torch.geometry2D import XY2Polar
from utils_torch.utils import EnsureFileDir

import matplotlib as mpl
from matplotlib import pyplot as plt

from World2D.Shapes2D import Shape2D

class Polygon(Shape2D):
    def __init__(self, param=None, data=None, **kw):
        super().__init__()
        utils_torch.model.InitForModel(self, param, data=data, ClassPath="World2D.Shapes.Polygon", **kw)
    def InitFromParam(self, param=None, IsLoad=False):
        if param is not None:
            self.param = param
        else:
            param = self.param
            data = self.data
            cache = self.cache
        cache.IsLoad = IsLoad
        cache.IsInit = not IsLoad
        cache.CollisionPoints = []
        cache.CollisionLines = [] 

        # Check Shape Type
        if GetAttrs(param, "Type") in ["Polygon", "Rectangle", "Triangle"]:
            SetAttrs(param.Type, "Polygon")
        else:
            raise Exception()
        EnsureAttrs(param, "Subtype", default="RegularPolygon")
        
        EnsureAttrs(param, "Internal", default="Inside")
        if param.Internal in ["Inside"]:
            self.IsInside = self.IsInsideShape
            self.IsOutside = self.IsOutsideShape
        else:
            self.IsInside = self.IsOutsideShape
            self.IsOutside = self.IsInsideShape
        
        if cache.IsInit:
            if HasAttrs(param, "Init.Center"):
                SetAttrs(param.Init, "Center.X", GetAttrs(param.Init.Center)[0])
                SetAttrs(param.Init, "Center.Y", GetAttrs(param.Init.Center)[1])

            EnsureAttrs(param, "Init.Method", default="FromVertices")
            if param.Init.Method in ["FromVertices", "FromVertex"]:
                self.CalculatEdgesFromVertices()
            elif param.Init.Method in ["CenterRadiusTheta"]:
                CheckAttrs(param.Subtype, value="RegularPolygon")
                self.InitRegularPolygon()
                self.CalculatEdgesFromVertices()
            else:
                raise Exception()
            data.VerticesNp = np.array(GetAttrs(param.Vertices), dtype=np.float32)
            self.CalculateEdgesInfo()
            self.CalculateBoundaryBox()
        #self.PlotShape(SavePath="./Polygon.png")
    def GenerateRandomInternalPositions(self):
        return
    def InitRegularPolygon(self):
        param = self.param
        Init = param.Init
        if not HasAttrs(param, "Edges.Num"):
            assert HasAttrs(param, "Vertices.Num")
            SetAttrs(param, "Edges.Num", param.Vertices.Num)        
        EnsureAttrs(param.Init, "")
        DirectionIncrement = math.pi * 2 / param.Edges.Num
        #PointsNp = np.zeros([param.Edges.Num, 2])
        Vertices = []
        EnsureAttrs(Init, "Rotation", default=0.0)
        DirectionCurrent = Init.Rotation
        for Index in range(param.Edges.Num):
            x, y = utils_torch.geometry2D.Polar2XY(Init.Radius, DirectionCurrent)
            Vertices.append([x + Init.Center.X, y + Init.Center.Y])
            DirectionCurrent += DirectionIncrement
        SetAttrs(param.Vertices, value=Vertices)
        return
    def IsOutsideShape(self, Points, MinDistance2Border=0.0): # coords:[point_num, (x, y)]. This method is only valid for convex polygon.
        Distance = self.Distance2Edges(Points) # [PointNum, EdgeNum]
        isOutside = np.sum(Distance < MinDistance2Border, axis=1) > 0
        return isOutside
    def IsInsideShape(self, Points, MinDistance2Border=0.0):
        return ~self.IsOutside(Points, MinDistance2Border) 
    def CalculatEdgesFromVertices(self):
        param = self.param
        EnsureAttrs(param.Vertices, "Num", len(GetAttrs(param.Vertices)))
        SetAttrs(param, "Edges", Vertices2VertexPairs(GetAttrs(param.Vertices)), Close=True)
        # VerticesNp = np.array(GetAttrs(param.Vertices), dtype=np.float32)
        # VerticesNp = np.concatenate((VerticesNp, VerticesNp[0][np.newaxis, :]), axis=0)
    def CalculateEdgesInfo(self):
        param = self.param
        data = self.data
        SetAttrs(param, "Edges.Num", len(param.Edges))
        SetAttrs(param.Edges, "Vector", utils_torch.geometry2D.VertexPairs2Vectors(GetAttrs(param.Edges)))
        SetAttrs(param.Edges, "Norm", utils_torch.geometry2D.Vectors2Norms(GetAttrs(param.Edges.Vector)))
        SetAttrs(param.Edges, "Norm.Direction", utils_torch.geometry2D.Vectors2Directions(GetAttrs(param.Edges.Norm)))
        data.EdgesNp = np.array(GetAttrs(param.Edges), dtype=np.float32)
        data.EdgesVectorNp = np.array(GetAttrs(param.Edges.Vector), dtype=np.float32)
        data.EdgesNormNp = np.array(GetAttrs(param.Edges.Norm), dtype=np.float32)
        data.EdgesNormDirectionNp = np.array(GetAttrs(param.Edges.Norm.Direction), dtype=np.float32)
    def CalculateBoundaryBox(self):
        param = self.param
        data = self.data
        # Calculate Boundary Box
        VerticesNp = np.array(GetAttrs(param.Vertices), dtype=np.float32) # [VertexNum, (x, y)]
        XMin = float(np.min(VerticesNp[:, 0]))
        XMax = float(np.max(VerticesNp[:, 0]))
        YMin = float(np.min(VerticesNp[:, 1]))
        YMax = float(np.max(VerticesNp[:, 1]))
        SetAttrs(param, "BoundaryBox", value=[XMin, YMin, XMax, YMax])
        SetAttrs(param, "BoundaryBox.XMin", XMin)
        SetAttrs(param, "BoundaryBox.XMax", XMax)
        SetAttrs(param, "BoundaryBox.YMin", YMin)
        SetAttrs(param, "BoundaryBox.YMax", YMax)
        SetAttrs(param, "BoundaryBox.Width", XMax - XMin)
        SetAttrs(param, "BoundaryBox.Height", YMax - YMin)
        SetAttrs(param, "BoundaryBox.Size", max(param.BoundaryBox.Width, param.BoundaryBox.Height))
        data.VerticesNp = VerticesNp
    def PrintInfo(self):
        utils_torch.AddLog('ArenaPolygon: edge_num:%d'%(self.edge_num))
        utils_torch.AddLog('center_coord:(%.1f, %.1f)'%(self.center_coord[0], self.center_coord[1]))
        utils_torch.AddLog('vertices:', end='')
        for index in range(self.edge_num):
            utils_torch.AddLog('(%.1f, %.1f)'%(self.vertices[index][0], self.vertices[index][1]), end='')
        utils_torch.AddLog('\n')
    def Vector2NearstBorder(self, Points):
        return
    def Distance2Edges(self, PointsNp):
        data = self.data
        return utils_torch.geometry2D.Distance2Edges(PointsNp, data.VerticesNp, data.EdgesNormNp)
    def IsMovingOutside(self, Points, NextPoints):
        return self.isInside(Points) * self.isOutside(NextPoints)
    def CheckCollision(self, XY, dXY):
        param = self.param
        data = self.data
        XYNext = XY + dXY
        PointNum = XY.shape[0]
        LambdasOfAllEdges = np.ones((PointNum, param.Edges.Num), dtype=np.float32)
        for Index in range(param.Edges.Num):
            LambdasOfAllEdges[:, Index] = utils_torch.geometry2D.InterceptRatio(XY, XYNext, data.EdgesNp[np.newaxis, Index, 0], data.EdgesNp[np.newaxis, Index, 1])        
        Lambdas = np.min(LambdasOfAllEdges, axis=1) # [PointNum, ShapeNum] --> [PointNum]
        CollisionPointIndices = np.argwhere(Lambdas<1.0) # [CollisionPointNum, 1] --> [CollisionPointNum]
        CollisionPointNum = CollisionPointIndices.shape[0]
        CollisionPointIndices = CollisionPointIndices.reshape(CollisionPointNum)
        CollisionEdgeIndices = np.argmin(LambdasOfAllEdges[CollisionPointIndices, :], axis=1)
        
        Collision = utils_torch.json.JsonObj2PyObj({
            "Num": CollisionPointNum,
            "Indices": CollisionPointIndices,
            "Lambdas": Lambdas[CollisionPointIndices], # [CollisionPointNum]
            "Norms": [], # [CollisionPointNum, Norms],
            "Edges": [],
        })

        if CollisionPointNum > 0:
            Collision.Norms = np.stack([data.EdgesNormNp[Index] for Index in CollisionEdgeIndices], axis=0)

        if CollisionPointNum > 0:
            Collision.Edges = np.stack([data.EdgesNp[Index] for Index in CollisionEdgeIndices], axis=0)
        self.LogCollision(XY[Collision.Indices], dXY[Collision.Indices], 
            Collision.Lambdas, Collision.Norms, Collision.Edges
        )
        return Collision
    def LogCollision(self, XYs, dXYs, Lambdas, Norms, Edges, SavePath="./CollisionPlot-0.png", Plot=False):
        CollisionNum = XYs.shape[0]
        cache = self.cache
        #EnsureAttrs(cache, "CollisionPoints", default=[])
        if CollisionNum > 0:
            if Plot:
                fig, axes = utils_torch.plot.CreateFigurePlt(CollisionNum)
            for Index in range(CollisionNum):
                Lambda = Lambdas[Index]
                XY, dXY = XYs[Index], dXYs[Index]
                Norm = Norms[Index]
                Edge = Edges[Index]
                cache.CollisionPoints.append(XY + Lambda * dXY)
                cache.CollisionLines.append([XY, XY + dXY])
                if Plot:
                    ax = utils_torch.plot.GetAx(axes, Index)
                    utils_torch.plot.PlotLineAndMarkVerticesXY(ax, Edge[0], Edge[1])
                    utils_torch.plot.PlotArrowAndMarkVerticesXY(ax, XY, XY + dXY)
                    utils_torch.plot.PlotPointAndAddText(ax, XY + Lambda * dXY, Text="IntersectionPoint")
                    utils_torch.plot.PlotDirectionOnEdge(ax, Edge, Norm)
                    utils_torch.plot.SetHeightWidthRatio(ax, 1.0)
                    ax.set_title("Lambda=%.2f"%Lambda)
            if Plot:
                plt.savefig(utils_torch.files.RenameIfPathExists(SavePath))
                plt.close()
    def PlotEdgesPlt(self, ax): # img: [H, W, C], np.uint8.
        utils_torch.plot.PlotLines(ax, self.EdgesNp)
    def PlotShape(self, ax=None, PlotNorm=False, PlotVertices=False, Save=True, SavePath="./", SetXYRange=True):
        param = self.param
        data = self.data
        if ax is None:
            plt.close("all")
            fig, ax = plt.subplots()
        utils_torch.plot.PlotPolyLine(ax, GetAttrs(param.Vertices))
        if PlotVertices:
            utils_torch.plot.PlotXYs(ax, GetAttrs(param.Vertices))
        
        if PlotNorm:
            utils_torch.plot.PlotDirectionsOnEdges(ax, GetAttrs(param.Edges), GetAttrs(param.Edges.Norm), Color="Red")
            #utils_torch.plot.PlotArrows(ax, utils_torch.geometry2D.Edges2MidPoints(GetAttrs(param.Edges)), GetAttrs(param.Edges.Norm), Color="Red")
            utils_torch.plot.PlotXYs(ax, utils_torch.geometry2D.Edges2MidPointsNp(data.EdgesNp) + data.EdgesNormNp, GetAttrs(param.Edges.Norm))
            
        if SetXYRange:
            ax.set_xlim(param.BoundaryBox.XMin, param.BoundaryBox.XMax)
            ax.set_ylim(param.BoundaryBox.YMin, param.BoundaryBox.YMax)
            utils_torch.plot.SetXTicksFloat(ax, param.BoundaryBox.XMin, param.BoundaryBox.XMax,)
            utils_torch.plot.SetYTicksFloat(ax, param.BoundaryBox.YMin, param.BoundaryBox.YMax,)
            ax.set_aspect(1)

        if Save:
            EnsureFileDir(SavePath)
            plt.savefig(SavePath, format="svg")
    
__MainClass__ = Polygon
utils_torch.model.SetMethodForWorldClass(__MainClass__)