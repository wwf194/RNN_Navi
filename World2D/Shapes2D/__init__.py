import World2D
from World2D.Shapes2D.Shape2D import Shape2D
from World2D.Shapes2D.Polygon import Polygon

def BuildShape(param):
    if param.Type in ["Polygon", "Polygon2D"]:
        return World2D.Shapes2D.Polygon(param)
    else:
        raise Exception()