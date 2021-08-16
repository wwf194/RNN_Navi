import pickle
import sys
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

<<<<<<< HEAD
from Models.place_cells import PlaceCells
=======
from Place_Cells import Place_Cells
>>>>>>> 2ced3c8656146ce297dcc8ba68cfd5f4dffd9f6f
#from Neurons_LIF import Neurons_LIF

class Linear_Navi(nn.Module):
    def __init__(self, dict_=None, load=False, f=None):
        super(Linear_Navi, self).__init__()
        if load:
            self.dict=torch.load(f, map_location=device) 
            self.i_0 = self.dict["i_0"] #input weight
            self.f = self.dict["f"] #output weight
            if(self.dict["task"] in ["pc", "pc_coords"]):
                self.place_cells = PlaceCells(load=True, dict_ = self.dict["place_cells"])
                self.place_cells_num = self.dict["place_cells"]["cell_num"]
        else:
            self.dict = dict_
            if(self.dict["task"] in ["pc", "pc_coords"]):
                self.place_cells = PlaceCells(dict_ = self.dict["place_cells"])
                self.place_cells_num = self.dict["place_cells"]["cell_num"]

            if task in ["pc"]:
                d0 = self.place_cells_num
            elif task in ["coords"]:
                d0 = self.dict["input_num"]
            self.i_0 = torch.nn.Parameter(torch.zeros((d0, self.dict["N_num"]), device=device, requires_grad=True))
            init_weight(self.i_0, self.dict["init_weight"]["i_0"])
            self.dict["i_0"] = self.i_0

            self.f = torch.nn.Parameter(torch.zeros((self.dict["N_num"], d0), device=device, requires_grad=True))
            torch.nn.init.normal_(self.f, 0.0, 1.0e-3 / self.dict["N_num"])
            init_weight(self.f, self.dict["init_weight"]["f"])             
            self.dict["f"] = self.f
            print(self.f)
            print(self.i_0)
            #input()

        self.task = self.dict["task"]
        if(self.task=="pc"):
            self.get_loss = self.get_loss_pc
            self.loss_list = {"pc":0.0, "act":0.0, "weight":0.0}
        elif(self.task=="coords"):
            self.get_loss = self.get_loss_coords
            self.loss_list = {"coords":0.0, "act":0.0, "weight":0.0}
        elif(self.task=="pc_coords"):
            self.get_loss = self.get_loss_pc_coords
            self.loss_list = {"pc":0.0, "coords":0.0, "act":0.0, "weight":0.0}
        else:
            print("invalid task:"+str(self.task))
            input()
        if(self.task in ["pc", "pc_coords"]):
            self.dict["pc_error"] = 0.0

        if("i_0" in self.dict["positive_weight"]):
            self.get_i_0 = lambda :torch.abs(self.i_0)
        else:
            self.get_i_0 = lambda :self.i_0
        if("f" in self.dict["positive_weight"]):            
            self.get_f = lambda :torch.abs(self.f)
        else:
            self.get_f = lambda :self.f
        
        self.loss_count = 0
        self.place_cells_act_cache = None
        self.act_func = get_act_func(self.dict["act_func"])
    def forward(self, inputs):
        x_0 = inputs[1]
        x_0 = x_0.to(device)#(batch_size, 2)
        place_cells_act = torch.squeeze(self.place_cells.get_activation(torch.unsqueeze(x_0,1)).float())#(batch_size, place_cells_num)
        self.place_cells_act_cache = place_cells_act
        cell_state = torch.mm(place_cells_act, self.get_i_0())#(batch_size, N_num)
        act = self.act_func(cell_state)
        output = torch.mm(act, self.get_f())#(batch_size, place_cells_num)
        return output, act
    def reset_loss(self):
        #print("aaa")
        self.loss_count = 0
        #print(self.loss_count)
        for key in self.loss_list.keys():
            self.loss_list[key] = 0.0
    def report_loss(self):
        for key in self.loss_list.keys():
            #print("%s:%s"%(key, str(self.loss_list[key]/self.loss_count)), end=" ")
            print("%s:%.4e"%(key, self.loss_list[key]/self.loss_count), end=" ")
        print("\n")
    def get_loss_pc(self, inputs, outputs_0):
        output, act = self.forward(inputs)
        self.dict["act_avg"] = torch.mean(torch.abs(act))
        pc_output = self.place_cells_act_cache
        self.dict["pc_error"] = ( torch.sum(torch.abs(output - pc_output)) / torch.sum(torch.abs(pc_output)) ).item() #relative place cells prediction error
        if(self.dict["pc_loss"]=="MSE"):
            loss_pc = index["pc"] * F.mse_loss(output, pc_output, reduction='mean')
        else: #Cross entropy loss
            output = F.softmax(output, dim=1) #(batch_size, place_cells_num)
            #loss_pc = - torch.mean( pc_output * torch.log(output) + (1.0 - pc_output) * torch.log(1.0-output) )
            loss_pc = F.binary_cross_entropy(input=output, target=pc_output)
        self.dict["pc_avg"] = torch.mean(torch.abs(pc_output))
        loss_act = act_cons_index * torch.mean(act ** 2)
        loss_weight = weight_cons_index * ( torch.mean(self.i_0 ** 2) + torch.mean(self.f ** 2) )
        self.loss_list["weight"] = self.loss_list["weight"] + loss_weight.item()
        self.loss_list["act"] = self.loss_list["act"] + loss_act.item()
        self.loss_list["pc"] = self.loss_list["pc"] + loss_pc.item()
        self.loss_count += 1
        return loss_pc + loss_act + loss_weight        
    def save(self, net_path):
        f = open(net_path+"state_dict.pth", "wb")
        net = self.to(torch.device("cpu"))
        torch.save(net.dict, f)
        net = self.to(device)
        f.close()