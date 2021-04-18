# -*- coding: utf-8 -*-
#import tensorflow as tf
import torch
import os
import math
import random
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import cv2 as cv

import config_sys
config_sys.set_sys_path()
from utils import get_from_dict, search_dict, get_name_args, ensure_path, contain
from utils_plot import *
from utils_plot import get_res_xy, plot_polyline_plt, get_int_coords_np

from anal_grid import get_score

fig, ax = plt.subplots()

im = ax.imshow(np.random.rand(50, 50))
ax_ = ax.inset_axes([1.05, 0.0, 0.12, 1.0])
print('aaa')
ax_.axis('off')
#ax_.add_patch(patches.Rectangle((0.0, 0.0), 1.0, 1.0, transform=plt.gca().transAxes, fill=False, edgecolor='blue'))
ax_.add_patch(patches.Rectangle((0.01, 0.01), 0.98, 0.98, fill=False, edgecolor='blue'))
print('bbb')
#ax_.add_patch(patches.Rectangle((0.0, 0.2), 1.0, 0.4, transform=plt.gca().transAxes, facecolor='blue'))
ax_.add_patch(patches.Rectangle((0.0, 0.2), 1.0, 0.4, facecolor='blue'))
print('bbb')
#ax_.add_line(Line2D([0.0, 1.0], [0.5, 0.5], transform=plt.gca().transAxes, color='red'))
ax_.add_line(Line2D([0.0, 1.0], [0.5, 0.5], color='red'))

plt.savefig('./anal/test_patch.png')
plt.close()