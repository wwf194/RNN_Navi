import random
import abc # asbtract method

import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import utils
from utils_torch.attrs import *
from utils_torch.plot import get_int_coords, get_int_coords_np, get_res_xy, get_float_coords_np

from utils.arena import *

import Environments

def InitFromParams(param):
    return Arenas2D(param)

def BuildArena(Param):
    if Param.Type in ["Polygon"]:
        return Environments.Polygon2D(Param)
    elif Param.Type in ["Circle"]:
        return Environments.ArenaCircle(Param)
    else:
        raise Exception('Arenas: Unsupported arena type: '+str(Param.Type))

class Arenas2D:
    def __init__(self, param=None):
        '''
        if options is not None:
            self.receive_options(options)
        else:
            raise Exception('Arenas: options must not be None.')
        '''
        if param is not None:
            self.InitFromParams(param)
    def InitFromParams(self, param):
        self.arenas = []
        for ArenaParam in param.Arenas:
            self.arenas.append(BuildArena(ArenaParam))
        self.set_current_arena(0)

    '''
    def plot_arenas_cv(self, save=True, save_path='./', **kw):
        for index, arena in enumerate(self.arenas):
            arena.plot_arena_cv(save=save, save_path=save_path, save_name='arena_%d_plot.png'%(index), **kw)
    '''
    def plot_arenas(self, save=True, save_path='./', **kw):
        for index, arena in enumerate(self.arenas):
            arena.plot_arena_plt(save=save, save_path=save_path, save_name='arena_%d_plot.png'%(index), **kw)

    def plot_random_xy_cv(self, res=50, save=True, save_path='./', **kw):
        for index, arena in enumerate(self.arenas):
            arena.plot_random_xy_cv(save=save, save_path=save_path, save_name='arena_%d_random_xy.png'%(index), **kw)

    def plot_random_xy(self, res=50, save=True, save_path='./', **kw):
        for index, arena in enumerate(self.arenas):
            arena.plot_random_xy_plt(save=save, save_path=save_path, save_name='arena_%d_random_xy.png'%(index), **kw)

    def receive_options(self, options):
        self.options = options

    def set_current_arena(self, index):
        self.arena_current = self.arenas[index]

    def get_current_arena(self):
        return self.arena_current
    
    def current_arena(self):
        return self.arena_current   

    def get_arena(self, index):
        return self.arenas[index]
