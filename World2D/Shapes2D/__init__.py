
import World2D
from World2D.Shapes2D.Shape2D import Shape2D
from World2D.Shapes2D.Polygon import Polygon
from World2D.Shapes2D.Circle import Circle

def BuildShape(param, LoadDir=None):
    if param.Type in ["Polygon", "Polygon2D"]:
        return World2D.Shapes2D.Polygon(param, LoadDir=LoadDir)
    elif param.Type in ["Circle", "Circle2D"]:
        return World2D.Shapes2D.Circle(param, LoadDir=LoadDir)
    else:
        raise Exception()