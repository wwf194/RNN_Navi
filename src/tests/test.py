import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR,ReduceLROnPlateau

import torchvision
from torchvision.datasets import mnist
import torchvision.transforms as transforms

import pickle

import time
import os

from param_config import *

from analysis import *
from analyze import quick_anal

from train_tools import evaluate, evaluate_iter, pytorch_info, prepare_CIFAR10
from PlaceCells import PlaceCells
from trajectory_generator import *
import cv2 as cv

#model to be trained.
from models import RNN_Navi

def main():
    #test_path()
    #anal_traj()
    #quick_anal()
    #net = load_net()
    #print(net.N.r)
    #test_path()
    #plot_place_cells_example()
    #plot_encoder_prediction()
    #plot_place_cells_prediction()
    #plot_traj()
    #net = load_net()
    #print(net.dict["i_0"])
    #print(torch.mean(net.dict["i_0"]))
    #print(net.dict["f"])
    #print(torch.mean(net.dict["f"]))

    array = np.array([1, 2, 3, 4, 5], dtype=np.float)
    print(array[1:-1])

def test_path():
    path = TrajectoryGenerator()
    traj = path.generate_trajectory(box_width=box_width, box_height=box_height, batch_size=batch_size, sample_num=sequence_length, random_init=(init_method=="input"))
    inputs, outputs = get_input_output(traj)

    inputs_cumu = 0.0
    for i in range(sequence_length):
        inputs_cumu += inputs[0, i]
        print(inputs[0, i], end="  ")
        print(outputs[0, i], end="  ")
        print(inputs_cumu)
    input()
    plot_num=10
    for i in range(plot_num):
        plot_traj(outputs_0[i], box_width=2.2, box_height=2.2, save_dir=save_dir_anal + "trajs/", save_name="trajectory_%d"%(i))

def print_performance(save_dir=save_dir_anal):
    net = load_net()
    
    #eval_net(net, logger)
    
    plot_training_curve(save_dir_stat+"train_statistics", loss_only=True)
    save_ratemaps(net=net, res=100, save_dir=save_dir)

    path = TrajectoryGenerator()
    net.eval()
    count = 0
    loss_total = 0.0
    traj = path.generate_trajectory(box_width=box_width, box_height=box_height, batch_size=batch_size, sample_num=sequence_length)
    inputs, outputs = get_input_output(traj)

    inputs=inputs.to(device)
    outputs_0 = outputs_0.to(device)
    outputs, act = net.forward(inputs) #act:(timestep, batch_size, N_num)
    #print("act shape:" + str(len(act)) + " " + str(len(act[0])) + str(list(act[0][0].size())))

    print("outputs_0:"+str(list(outputs_0.size())))
    print("outputs:"+str(list(outputs.size())))
    print(outputs)
    print(outputs_0)

    anal_num = 10
    for i in range(anal_num):
        batch_index = random(range(batch_size),1)[0]
        place_cell_index = random(range(place_cells_num),1)[0]
        for j in range(sequence_length):
           print(outputs[j][batch_index][place_cell_index])
        print(outputs_0[batch_index,:,place_cell_index])

if __name__ == '__main__':
    main()