import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR,ReduceLROnPlateau

import random

import matplotlib.pyplot as plt
plt.box(False)
from matplotlib.ticker import FuncFormatter 
import pandas as pd
import numpy as np
import pickle
import seaborn as sns

import scipy

from my_logger import my_logger
import time
import os
import math

from config import epoch_num, device, logger, device

import matplotlib as mpl
import importlib
importlib.reload(mpl); importlib.reload(plt); importlib.reload(sns)

# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt

import scipy.stats
from imageio import imsave
import cv2

from anal_functions import *
from param_config import *
from grid_scorer import GridScorer

print(model_type)

#from tqdm.autonotebook import tqdm

def main():
    #global_trajectory_generator=TrajectoryGenerator(arena_types=arena_types)
    #plot_traj(trajectory_generator=global_trajectory_generator, save_dir=save_dir_anal+"test/", plot_num=10)
    '''
    f0 = open(save_dir_anal+"analysis.txt",'w')
    logger=my_logger()
    logger.clear()
    logger.add_stdout()
    logger.add_flow(f0, name="log_file")
    quick_anal(net=None, logger=logger, save_dir = save_dir_anal+"test/")
    f0.close()
    '''
    #print(model_type)
    #model_type="lstm"
    net = load_net()
    net = net.to(device)
    #test_anal(net=net, logger=None, save_dir=save_dir_anal + "epoch_%d/"%(-2))
    #quick_anal(net=net, logger=None, save_dir=save_dir_anal + "epoch_%d/"%(-2))
    #print(net.encoder(torch.from_numpy(np.nd_array([0.5, 0.5]))), device=device)

    #plot_init_positions(save_dir = save_dir_anal+"init_positions/")

def anal_1(logger, save_dir=save_dir_anal):
    a=1
def eval_net(net, logger):
    logger.write("evaluating net performance")
    val_loss, val_acc=cal_iter_data(net)
    logger.write('val_loss:%.4f val_acc:%.4f'%(val_loss, val_acc))

def test_anal(net=None, logger=None, save_dir=save_dir_anal):
    EnsureDir(save_dir)
    if net is None:
        net = load_net()
        net = net.to(device)
    res=ratemap_res
    cmap = ['jet','bwr']
    """Scoring ratemaps given trajectories.
    Args:
      nbins: Number of bins per dimension in the ratemap.
      coords_range: Environment coordinates range.
      mask_parameters: parameters for the masks that analyze the angular
        autocorrelation of the 2D autocorrelation.
      min_max: Correction.
    """
    # Create scorer objects
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    masks_parameters = list(zip(starts, ends.tolist()))
    print(masks_parameters)
    #input()
    grid_scorer_ = GridScorer(nbins=res, mask_parameters=masks_parameters, min_max=True)

    for num in range(len(arena_types)):
        acts, masks = compute_ratemaps(net=net, res=res, random_init=random_init_, arena_index=num)
        rm_fig = plot_ratemaps(acts, n_plots=len(acts), exaggerate=False, cmap=cmap, masks=masks, arena_index=num)
        count = 0
        for key in rm_fig.keys():
            print(rm_fig[key].shape)
            #imsave(save_dir + "/" + "heatmap_%s_%s_%d"%(cmap[count], str(arena_types[num]), num) + ".png", rm_fig[key])
            figs = rm_fig[key]
            for fig in figs:
                grid_scores = grid_scorer_.get_scores(rate_map=fig)
                print(grid_scores)
            count += 1

def quick_anal(net=None, logger=None, save_dir=save_dir_anal):
    EnsureDir(save_dir)
    if net is None:
        net = load_net()
        net = net.to(device)

    res=ratemap_res

    if model_type=="linear":
        logger.write("i_0:" + get_data_stat(net.get_i_0().detach().cpu().numpy()))
        logger.write("f:" + get_data_stat(net.get_f().detach().cpu().numpy()))

    plot_spatial_pattern(net, res, save_dir)
    if model_type in ["rnn","linear"]:
        visualize_weight(net, name="r", save_dir=save_dir)

def full_anal(net=None, logger=None, save_dir=save_dir_anal):
    EnsureDir(save_dir)
    if(net is None):
        net = load_net()
        net = net.to(device)

    res=ratemap_res

    if(model_type=="linear"):
        logger.write("i_0:" + get_data_stat(net.get_i_0().detach().cpu().numpy()))
        logger.write("f:" + get_data_stat(net.get_f().detach().cpu().numpy()))

    plot_spatial_pattern(net, res, save_dir)

def plot_spatial_pattern(net, res, save_dir):
    cmap = ['jet','bwr']

    if global_trajectory_generator.dict.get("arena_types_test") is not None:
        global_cache["use_test_arenas"] = True
        #global_cache["print_init_position"] = True
        for num in range(len(arena_types_test)):
            acts, masks = compute_ratemaps(net=net, res=res, random_init=random_init_, arena_index=num)
            if net.dict["separate_ei"]:
                acts_ei = [acts[0:net.dict["N"]["E_num"],:,:], acts[net.dict["N"]["E_num"]:net.dict["N"]["N_num"],:,:]]
                names = ["_e", "_i"]
                count = 0
                for act in acts_ei:
                    count2 = 0
                    rm_fig = plot_ratemaps(act, n_plots=len(act), exaggerate=False, cmap=cmap, masks=masks, arena_index=num)
                    for key in rm_fig.keys():
                        print(rm_fig[key].shape)
                        imsave(save_dir + "/" + "heatmap_%s_%s_%s_%d(test)"%(cmap[count2], names[count], str(arena_types[num]), num) + ".png", rm_fig[key])
                        count2+=1
                    count += 1
            else:
                rm_fig = plot_ratemaps(acts, n_plots=len(acts), exaggerate=False, cmap=cmap, masks=masks, arena_index=num)
                count = 0
                for key in rm_fig.keys():
                    print(rm_fig[key].shape)
                    imsave(save_dir + "/" + "heatmap_%s_%s_%d(test)"%(cmap[count], str(arena_types[num]), num) + ".png", rm_fig[key])
                    count += 1
            if(net.dict["task"] in ["coords", "pc_coords"]):
                compare_traj(net=net, save_dir=save_dir, save_name="traj plot(%s-%d)(test)"%(str(arena_types[num]), num), arena_index=num)
        global_cache["use_test_arenas"] = False
        global_cache["print_init_position"] = False
    #plot_training_curve(save_dir_stat+"train_statistics", loss_only=True)
    for num in range(len(arena_types)):
        acts, masks = compute_ratemaps(net=net, res=res, random_init=random_init_, arena_index=num)
        if net.dict["separate_ei"]:
            acts_ei = [acts[0:net.dict["N"]["E_num"],:,:], acts[net.dict["N"]["E_num"]:net.dict["N"]["N_num"],:,:]]
            names = ["_e", "_i"]
            count = 0
            for act in acts_ei:
                count2 = 0
                rm_fig = plot_ratemaps(act, n_plots=len(act), exaggerate=False, cmap=cmap, masks=masks, arena_index=num)
                for key in rm_fig.keys():
                    print(rm_fig[key].shape)
                    imsave(save_dir + "/" + "heatmap_%s_%s_%s_%d"%(cmap[count2], names[count], str(arena_types[num]), num) + ".png", rm_fig[key])
                    count2+=1
                count += 1
        else:
            rm_fig = plot_ratemaps(acts, n_plots=len(acts), exaggerate=False, cmap=cmap, masks=masks, arena_index=num)
            count = 0
            for key in rm_fig.keys():
                print(rm_fig[key].shape)
                imsave(save_dir + "/" + "heatmap_%s_%s_%d"%(cmap[count], str(arena_types[num]), num) + ".png", rm_fig[key])
                count += 1
        if(net.dict["task"] in ["coords", "pc_coords"]):
            compare_traj(net=net, save_dir=save_dir, save_name="traj plot(%s-%d)"%(str(arena_types[num]), num), arena_index=num)
    
    if(net.dict["task"] in ["pc", "pc_coords"]):
        plot_encoder_prediction(net=net, res=res, save_dir=save_dir)
        plot_place_cells_prediction(net=net, res=res, save_dir=save_dir, acts=acts)

if __name__ == '__main__':
    main()