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

def BuildArena(param, **kw):
    return World2D.Arena2D(param, **kw)

class Arenas2D:
    def __init__(self, param=None, data=None, **kw):
        utils_torch.model.InitForModel(self, param, data, ClassPath="World2D.Arenas2D", **kw)
    def InitFromParam(self, param=None, IsLoad=False):
        if param is not None:
            self.param = param
        else:
            param = self.param
            cache = self.cache
        cache.IsLoad = IsLoad
        cache.Modules = utils_torch.EmptyPyObj()
        self.Arenas = []
        for Index, ArenaParam in enumerate(GetAttrs(param.Arenas)):
            Arena = BuildArena(ArenaParam, LoadDir=cache.LoadDir)
            Arena.InitFromParam(IsLoad=cache.IsLoad)
            self.Arenas.append(Arena)
            setattr(cache.Modules, "Arena%d"%Index, Arena)
        self.SetCurrentArena(0)
        self.PlotInsideMasks()
    def PlotInsideMasks(self, SaveDir=None):
        if SaveDir is None:
            SaveDir = utils_torch.GetMainSaveDir()
        for Index, Arena in enumerate(self.Arenas):
            Arena.PlotInsideMask(
                Save=True,
                SavePath=utils_torch.GetMainSaveDir() + "Arenas/" + "Arenas2D-InsideMask=%d.png"%Index
            )
    def PlotArenas(self, save=True, save_path='./', **kw):
        for index, arena in enumerate(self.Arenas):
            arena.Plotarena_plt(save=save, save_path=save_path, save_name='arena_%d_plot.png'%(index), **kw)
    def PlotCurrentArena(self, ax=None, Save=False):
        self.GetCurrentArena().PlotArena(ax, Save)
    def SetCurrentArena(self, Index):
        self.ArenaCurrent = self.Arenas[Index]
    def GetCurrentArena(self):
        return self.ArenaCurrent
    def GetArenaByIndex(self, Index):
        return self.Arenas[Index]
    def GetArena(self, index):
        return self.Arenas[index]
    # def SetFullName(self, Name):
    #     param = self.param
    #     param.FullName = Name
    #     for Index, Arena in enumerate(self.Arenas):
    #         Arena.SetFullName(Name + ".Arena%d"%Index)
    # def Save(self, SaveDir, IsRoot=True):
    #     param = self.param
    #     if IsRoot:
    #         utils_torch.json.PyObj2JsonFile(param, SaveDir + param.FullName + ".param.jsonc")
    #     for Arena in self.Arenas:
    #         Arena.Save(SaveDir, IsRoot=False)
__MainClass__ = Arenas2D
utils_torch.model.SetMethodForWorldClass(__MainClass__)