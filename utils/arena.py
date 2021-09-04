import random
import math
from enum import IntEnum
import numpy as np
from utils_torch.attrs import EnsureAttrs
from utils_torch.geometry2D import Polar2XY, XY2Polar, XY2PolarNp


class LatticeState(IntEnum):
    Undefined = 0,
    Internal = 1,
    External = 2,
    Border = 3,
    BoundaryBoxBorder = 5,

class Direction(IntEnum):
    UpLeft = 0,
    Up = 1,
    UpRight = 2,
    Left = 3,
    Right = 4,
    DownLeft = 5,
    Down = 6,
    DownRight = 7,
    Null = 8

class SquareLattice:
    def __init__(self, param=None):
        if param is not None:
            self.param = param
    def InitFromParam(self, param):
        self.param = param
        self.LatticeState = np.array()
        EnsureAttrs(param, "Initialize", default=[])
        for Order in param.Initialize:
            self.ImplementOrder(Order)
    def ImplementOrder(self, Order):
        return
    def SetOutsideAsExternal(self, Points, ExternalPointExample=None):
        return
    def SetInsideAsInternal(self, Points):
        return

def FloodFill(Lattice, StartPointIndex, State, BoundaryState, ):
    _FloodFill(Lattice, StartPointIndex, State, BoundaryState, Direction.Null)

def _FloodFill(Lattice, PointIndex, State, BoundaryState, ParentDirection):
    for NeightborPointIndex in GetNeighborPointsNonCorner(PointIndex, ParentDirection):
        _SetLatticeState(Lattice, NeightborPointIndex, State, BoundaryState)
    [UpLeft, UpRight, DownLeft, DownRight] = GetNeighborPointsCorner(PointIndex)
    if ParentDirection != Direction.UpLeft and Lattice[UpLeft] not in BoundaryState:
        _FloodFill(Lattice, PointIndex, State, BoundaryState, Direction.DownRight)
    if ParentDirection != Direction.UpRight and Lattice[UpRight] not in BoundaryState:
        _FloodFill(Lattice, PointIndex, State, BoundaryState, Direction.DownLeft)
    if ParentDirection != Direction.DownLeft and Lattice[DownLeft] not in BoundaryState:
        _FloodFill(Lattice, PointIndex, State, BoundaryState, Direction.UpRight)
    if ParentDirection != Direction.DownRight and Lattice[DownRight] not in BoundaryState:
        _FloodFill(Lattice, PointIndex, State, BoundaryState, Direction.UpLeft)      

def _SetLatticeState(Lattice, PointIndex, State, BoundaryState):
    if Lattice[PointIndex] not in BoundaryState:
        Lattice[PointIndex] = State

def GetNeighborPointsNonCorner(PointIndex, ParentDireciton=Direction.Null): # 上下左右
    xIndex, yIndex = PointIndex[0], PointIndex[1]
    if ParentDireciton == Direction.Null:
        return [
            (xIndex - 1, yIndex),
            (xIndex, yIndex - 1),
            (xIndex, yIndex + 1),
            (xIndex + 1, yIndex),
        ]
    elif ParentDireciton == Direction.UpLeft:
        return [
            (xIndex, yIndex - 1),
            (xIndex + 1, yIndex),
        ]
    elif ParentDireciton == Direction.UpRight:
        return [
            (xIndex - 1, yIndex),
            (xIndex, yIndex - 1),
        ]
    elif ParentDireciton == Direction.DownLeft:
        return [
            (xIndex + 1, yIndex),
            (xIndex, yIndex + 1),
        ]
    elif ParentDireciton == Direction.DownRight:
        return [
            (xIndex - 1, yIndex),
            (xIndex, yIndex + 1),
        ]
    else:
        raise Exception()

GetNeighborPointsUpDownLeftRight = GetNeighborPointsNonCorner

def GetNeighborPointsCorner(PointIndex): # 左上，右上，左下，右下
    xIndex, yIndex = PointIndex[0], PointIndex[1]
    return [
        (xIndex - 1, yIndex + 1),
        (xIndex + 1, yIndex + 1),
        (xIndex - 1, yIndex - 1),
        (xIndex + 1, yIndex - 1),
    ]

def GetNeighborPoints(PointIndex):
    xIndex, yIndex = PointIndex[0], PointIndex[1]
    return [
        (xIndex - 1, yIndex - 1),
        (xIndex - 1, yIndex),
        (xIndex - 1, yIndex + 1),
        (xIndex, yIndex - 1),
        (xIndex, yIndex + 1),
        (xIndex + 1, yIndex - 1),
        (xIndex + 1, yIndex),
        (xIndex + 1, yIndex + 1),
    ]

def GetNeighborPointsUpLeft(xIndex, yIndex):
    return [
        (xIndex - 1, yIndex - 1),
        (xIndex - 1, yIndex),
        (xIndex - 1, yIndex + 1),
        (xIndex, yIndex - 1),
        (xIndex, yIndex + 1),
        (xIndex + 1, yIndex - 1),
        (xIndex + 1, yIndex),
        (xIndex + 1, yIndex + 1),
    ]