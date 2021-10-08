import numpy as np

import utils_torch
from utils_torch.attrs import *

import World2D.Shapes2D as Shapes2D
class Circle(Shapes2D.Shape2D):
    def __init__(self, param=None, data=None, **kw):
        super().__init__()
        utils_torch.model.InitForModel(self, param, data, ClassPath="World2D.Shapes2D.Circle", **kw)
    def InitFromParam(self, param=None, IsLoad=False):
        if param is not None:
            self.param = param
        else:
            param = self.param
            cache = self.cache

        cache.IsLoad = IsLoad
        cache.IsInit = not IsLoad
        if cache.IsInit:
            CheckAttrs(param, "Type", value="Circle")
            EnsureAttrs(param, "Init.Method", default="CenterRadius")
            if param.Init.Method in ["CenterRadius"]:
                if not HasAttrs(param, "Center"):
                    param.Center = GetAttrs(param.Init, "Center")
                if not HasAttrs(param, "Radius"):
                    param.Radius = GetAttrs(param.Init, "Radius")
            else:
                raise Exception()
            SetAttrs(param, "Center.X", GetAttrs(param.Center)[0])
            SetAttrs(param, "Center.Y", GetAttrs(param.Center)[1])
            self.CalculateBoundaryBox()
        cache.Center = np.array(GetAttrs(param.Center), dtype=np.float)
    def CalculateBoundaryBox(self):
        param = self.param
        # Calculate Boundary Box
        if not HasAttrs(param, "BoundaryBox"):
            XMin = param.Center.X - param.Radius
            XMax = param.Center.X + param.Radius
            YMin = param.Center.Y - param.Radius
            YMax = param.Center.Y + param.Radius
            SetAttrs(param, "BoundaryBox", [XMin, YMin, XMax, YMax])
            SetAttrs(param, "BoundaryBox.XMin", XMin)
            SetAttrs(param, "BoundaryBox.XMax", XMax)
            SetAttrs(param, "BoundaryBox.YMin", YMin)
            SetAttrs(param, "BoundaryBox.YMax", YMax)
            SetAttrs(param, "BoundaryBox.Width", XMax - XMin)
            SetAttrs(param, "BoundaryBox.Height", YMax - YMin)
            SetAttrs(param, "BoundaryBox.Size", max(param.BoundaryBox.Width, param.BoundaryBox.Height))
    def Distance2Center(self, Points):
        cache = self.cache
        Vector2Center = cache.Center[np.newaxis, :] - Points # [PointNum, (x, y)]
        Distance2Center = np.linalg.norm(Vector2Center, axis=-1) # [PointNum]
        return Distance2Center
    def DistanceAndDirectionFromCenter(self, XYs): # xy: [PointNum, 2]
        cache = self.cache
        VectorFromCenter2XYs = XYs - cache.Center[np.newaxis, :] # [PointNum, (x, y)]
        Distance = np.linalg.norm(VectorFromCenter2XYs, ord=2, axis=-1) # [PointNum]
        Direction = np.arctan2(XYs[:, 1], XYs[:, 0])
        return Distance, Direction
    def IsOutside(self, XYs, MinDistance2Border=0.0): # Points:[PointNum, (x, y)]. This method is only valid for convex polygon.
        return self.Distance2Center(XYs) > (self.param.Radius + MinDistance2Border)
    def IsInside(self, XYs, MinDistance2Border=0.0):
        return self.Distance2Center(XYs) + MinDistance2Border < self.param.Radius
    def Vector2NearstBorder(self, XYs):
        return
    def ReportInfo(self):
        param = self.param
        utils_torch.AddLog("Circle2D: ", end="")
        utils_torch.AddLog("Center:(%.1f, %.1f)"%(param.Center.X, param.Center.Y))
        utils_torch.AddLog("Radius:(%.3f)"%param.Radius)

__MainClass__ = Circle
utils_torch.model.SetMethodForWorldClass(__MainClass__)