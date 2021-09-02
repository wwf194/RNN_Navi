# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import torch
import scipy
from param_config import *
import torch.nn.functional as F

class ring_stimuli(object):
    def __init__(self, dict_, load=False):
        self.dict = dict_
        if load:
            self.unit_num = self.dict["unit_num"]
            self.sigma = self.dict["sigma"]
            self.peak = self.dict["peak"]
            self.dist = self.dict["dist"]
            self.box_width = self.dict["box_width"]
            self.box_height = self.dict["box_height"]
            self.coords = self.dict["coords"]
            self.ratio = self.dict["ratio"]
        else:
            self.unit_num = dict
            if self.dict["start_position"]=="zero"
                self.theta_0 = 0.0
            elif self.dict["start_position"]=="random":
                self.theta_0 = random.random() * np.pi * 2 / self.dict["unit_num"] #random.random() generates a random float in (0,1)

        #self.sigma = self.

        self.ratio_square = self.ratio * self.ratio
        self.norm_local = self.dict["norm_local"]
        self.positive = self.dict["positive"]
        self.separate_softmax = self.dict["separate_softmax"]
    
        
        self.unit_interval = np.pi * 2 / self.unit_num
        unit_positions = np.array([unit_index * ni.pi * 2 /self.unit_num for unit_index in range(self.unit_num)])
        unit_positions = unit_positions + self.theta_0
        unit_positions = np.mod(unit_positions + np.pi, np.pi * 2) - np.pi #range: (-pi, pi)
        self.unit_positions = unit_positions

        

    def Getactivation(self, direct): #direct:(batch_size), range:(-pi, pi)
        
        dists = ( self.unit_positions - direct )


        direct = np.mod(direct + np.pi, np.pi * 2) - np.pi #range:(-pi, pi)




    def Getactivation_single(self, pos): #pos:(batch_size, sequence_length, (x,y))
        expand_pos = torch.unsqueeze(pos, dim=2)
        #print(list(expand_pos.size()))
        expand_coords = torch.unsqueeze(torch.unsqueeze(self.coords, dim=0), dim=0)
        #print(list(expand_coords.size()))
        vec = expand_pos - expand_coords#(batch_size, sequence_length, place_cell_num, (x,y))
        dist = torch.sum(vec ** 2, dim=3) #(batch_size, sequence_length, place_cell_num, 1)
        act = torch.exp(- dist / (2 * self.sigma * self.sigma))
        if(self.norm_local == True):
            act /= torch.sum(act, dim=2, keepdim=True)         
        act = torch.squeeze(act)
        return pc_act_index * act #(batch_size, sequence_length, act)