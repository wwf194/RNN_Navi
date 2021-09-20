import random
import abc # asbtract method

import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import utils
from utils_torch.attrs import *
#from utils_torch.plot import Getint_coords, Getint_coords_np, Getres_xy, Getfloat_coords_np

from utils.arena import *

import World2D

def InitFromParam(param):
    return Arenas2D(param)

def BuildArena(param):
    return World2D.Arena2D(param)

class Arenas2D:
    def __init__(self, param=None):
        '''
        if options is not None:
            self.receive_options(options)
        else:
            raise Exception('Arenas: options must not be None.')
        '''
        if param is not None:
            self.param = param
    def InitFromParam(self, param=None):
        if param is not None:
            self.param = param
        else:
            param = self.param
        self.Arenas = []
        for ArenaParam in param.Arenas:
            Arena = BuildArena(ArenaParam)
            Arena.InitFromParam()
            self.Arenas.append(Arena)
        self.SetCurrentArena(0)
        for Index, Arena in enumerate(self.Arenas):
            Arena.PlotInsideMask(Save=True, SavePath=utils.ArgsGlobal.SaveDir + "Arenas2D-InsideMask=%d.png"%Index)
    '''
    def PlotArenas_cv(self, save=True, save_path='./', **kw):
        for index, arena in enumerate(self.Arenas):
            arena.Plotarena_cv(save=save, save_path=save_path, save_name='arena_%d_plot.png'%(index), **kw)
    '''
    def PlotArenas(self, save=True, save_path='./', **kw):
        for index, arena in enumerate(self.Arenas):
            arena.Plotarena_plt(save=save, save_path=save_path, save_name='arena_%d_plot.png'%(index), **kw)
    def Plotrandom_xy_cv(self, res=50, save=True, save_path='./', **kw):
        for index, arena in enumerate(self.Arenas):
            arena.Plotrandom_xy_cv(save=save, save_path=save_path, save_name='arena_%d_random_xy.png'%(index), **kw)
    def Plotrandom_xy(self, res=50, save=True, save_path='./', **kw):
        for index, arena in enumerate(self.Arenas):
            arena.Plotrandom_xy_plt(save=save, save_path=save_path, save_name='arena_%d_random_xy.png'%(index), **kw)
    def SetCurrentArena(self, Index):
        self.ArenaCurrent = self.Arenas[Index]
    def GetCurrentArena(self):
        return self.ArenaCurrent
    def GetArenaByIndex(self, Index):
        return self.Arenas[Index]
    def GetArena(self, index):
        return self.Arenas[index]

__MainClass__ = Arenas2D