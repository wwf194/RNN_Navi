import numpy as np

class Shape2D:
    def __init__(self, param=None):
        return
    def InitFromParam(self, param=None):
        if param is not None:
            self.param = param
        else:
            param = self.param
    def GetInsideMask(self, BoundaryBox, ResolutionX, ResolutionY):
        mask = np.zeros(ResolutionX, ResolutionY)


