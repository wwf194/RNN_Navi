import random
import math

import numpy as np
from utils_torch.math import polar2xy, xy2polar, xy2polarNp
from utils_torch import get_from_dict, search_dict

def Vertices2VertexPairs(Vertices, close=True):
    VertexNum = len(Vertices)
    VertexPairs = []
    if close:
        for Index in VertexNum:
            VertexPairs.append(Vertices[Index], Vertices[(Index + 1) % VertexNum])
    return

def Vertices2Vectors(Vertices, close=True): 
    return Vertices2EdgesNp(np.array(Vertices, dtype=np.float32), close=close).tolist()

def Vertices2EdgesNp(VerticesNp, close=True):
    if close:
        VerticesNp = np.concatenate((VerticesNp, VerticesNp[0,:][np.newaxis, :]), axis=0)
    VectorsNp = np.diff(VerticesNp, axis=0)
    return VectorsNp

def VertiexPairs2Vectors(VertexPairs):
    Vectors = []
    for VertexPair in VertexPairs:
        Vectors.append(VertexPair2Vector(VertexPair))
    return Vectors

def VertexPair2Vector(VertexPair):
    return [[VertexPair[0][1] - VertexPair[0][0]], VertexPair[1][1] - VertexPair[1][0]]

def VertexPair2VectorNp(VertexPairNp): # ((x0, y0), (x1, y1))
    return np.diff(VertexPairNp, axis=1)

def Vectors2Norms(Vectors):
    return Vectors2NormsNp(np.array(Vectors, dtype=np.float32)).tolist()

def Vectors2NormsNp(VectorsNp):  # Calculate Norm Vectors Pointing From Inside To Outside Of Polygon
    #VectorNum = len(Vectors)  
    #Vectors = np.array(Vectors, dtype=np.float32)
    VectorNum = VectorsNp.shape[0]
    VectorsNorm = np.zeros([VectorNum, 2])
    # (a, b) is vertical to (b, -a)
    VectorsNorm[:, 0] = VectorsNp[:, 1]
    VectorsNorm[:, 1] = - VectorsNp[:, 0]
    # Normalize To Unit Length
    VectorsNorm = VectorsNorm / (np.linalg.norm(VectorNorms, axis=1, keepdims=True))
    return VectorsNorm

def Vectors2DirectionsNp(Vectors):
    Directions = []
    Directions = xy2polarNp(Vectors)
    return Directions

def Vectors2Directions(Vectors):
    Directions = []
    for Vector in Vectors:
        Directions.append(xy2polar(Vector))
    return Directions

def Distance2Edges(PointsNp, EdgeVerticesNp, EdgeNormsNp):
    Points2EdgeVertices = EdgeVerticesNp[:, np.newaxis, :] - PointsNp[np.newaxis, :, :] # [1, VerticesNum, 2] - [PointNum, 1, 2] = [PointsNum, VerticesNum, 2]
    return np.sum(Points2EdgeVertices * EdgeNormsNp[np.newaxis, :, :], axies=-1)