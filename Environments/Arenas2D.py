import random
import abc # asbtract method

import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils.plot import *
from utils.plot import get_int_coords, get_int_coords_np, get_res_xy, get_float_coords_np
import utils
from utils_torch.attrs import *

from utils.arena import *

import Environments

def InitFromParams(param):
    return Arenas2D(param)

def build_arena(param):
    if param.type in ["Polygon"]:
        return Environments.ArenaPolygon(param)
    elif param.type in ["Circle"]:
        return Environments.ArenaCircle(param)
    else:
        raise Exception()

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
        for arenaParam in param.arenas:
            self.arenas.append(build_arena(arenaParam))
        self.set_current_arena(0)

    def build_arena(self, arena_dict, load):
        type_ = arena_dict['type']
        if type_ in ['sqaure', 'polygon', 'square_max', 'rec', 'rectangle', 'rec_max'] or isinstance(type_, int):
            return Arena_Polygon(arena_dict, load=load)    
        elif type_ in ['circle']:
            return Arena_Circle(arena_dict, load=load)
        else:
            raise Exception('Arenas: Unsupported arena_type: '+str(type_))
    
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
