import random
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

#training parameters.
from utils_anal import *
from Place_Cells import Place_Cells

class LSTM_Navi(nn.Module):
    def __init__(self, dict_=None, load=False, f=None):
        super(LSTM_Navi, self).__init__()
        if load:
            self.dict = torch.load(f, map_location=device)
            #self.dict = pickle.load(f)
            #print(self.dict)
            #print(self.dict.keys())
            self.lstm = nn.LSTM(input_size=self.dict["input_num"], hidden_size=self.dict["N_num"], batch_first=True)
            self.lstm.load_state_dict(self.dict["lstm_dict"])
            self.f = self.dict["f"]
            init_method = get_name(self.dict["init_method"])
            if init_method in ["input", "linear"]:
                self.i_0_h = nn.Parameter(self.dict["i_0_h"])
                self.i_0_c = nn.Parameter(self.dict["i_0_c"])
            elif init_method in ["mlp"]:
                print(isinstance(self.dict["encoder_dict"], dict))
                self.encoder = load_mlp(self.dict["encoder_dict"])
                self.add_module("encoder", self.encoder)
            elif init_method in ["fixed"]:
                self.h_0 = nn.Parameter(self.dict["h_0"])
                self.c_0 = nn.Parameter(self.dict["c_0"])
            if self.dict["task"] in ["pc", "pc_coords"]:
                self.place_cells = PlaceCells(load=True, dict_ = self.dict["place_cells"])
                self.place_cells_num = self.dict["place_cells"]["cell_num"]
        else:
            self.dict = dict_
            self.lstm = nn.LSTM(input_size=self.dict["input_num"], hidden_size=self.dict["N_num"], batch_first=True)
            self.f = torch.nn.Parameter(torch.zeros((self.dict["N_num"], self.dict["output_num"]), device=device, requires_grad=True)) #input weights
            init_weight(self.f, self.dict["init_weight"]["f"])
            self.dict["f"] = self.f
        
            if self.dict["task"] in ["pc", "pc_coords"]:
                self.place_cells = PlaceCells(dict_ = self.dict["place_cells"])
                self.place_cells_num = self.dict["place_cells"]["cell_num"]

            init_method = get_name(self.dict["init_method"])
            if init_method in ["linear", "input", "mlp"]:
                if(task in ["pc"]):
                    input_dim = self.place_cells_num
                elif(task in ["coords"]):
                    input_dim = self.dict["output_num"]
            if init_method in ["linear", "input"]:
                self.i_0_h = torch.nn.Parameter(torch.zeros((input_dim, self.dict["N_num"]), device=device, requires_grad=True))
                self.i_0_c = torch.nn.Parameter(torch.zeros((input_dim, self.dict["N_num"]), device=device, requires_grad=True))
                init_weight(self.i_0_h, self.dict["init_weight"]["i_0"])
                init_weight(self.i_0_c, self.dict["init_weight"]["i_0"])
                self.dict["i_0_h"] = self.i_0_h
                self.dict["i_0_c"] = self.i_0_c

                if("i_0_h" in positive_weight):
                    self.get_i_0_x = lambda :torch.abs(self.i_0_x)
                else:
                    self.get_i_0_x = lambda :self.i_0_x
                if("i_0_c" in positive_weight):
                    self.get_i_0_r = lambda :torch.abs(self.i_0_r)
                else:
                    self.get_i_0_r = lambda :self.i_0_r
            elif init_method=="mlp":
                self.encoder, self.dict["encoder_dict"] = build_mlp(get_arg(self.dict["init_method"]))
                #print(isinstance(self.dict["encoder_dict"], dict))
                #print(get_arg(self.dict["init_method"])["layer_dicts"])
                #input()
                self.add_module("encoder", self.encoder)
            elif init_method=="fixed":
                self.h_0 = torch.nn.Parameter(torch.zeros((self.dict["N_num"]), device=device, requires_grad=True))
                self.c_0 = torch.nn.Parameter(torch.zeros((self.dict["N_num"]), device=device, requires_grad=True))
                init_weight(self.h_0, self.dict["init_weight"]["i_0"])
                init_weight(self.c_0, self.dict["init_weight"]["i_0"])                
                self.dict["h_0"] = self.h_0
                self.dict["c_0"] = self.c_0

            self.dict["lstm_dict"] = self.lstm.state_dict()
            


        self.add_module("lstm", self.lstm)

        self.get_f = lambda :self.f
        
        self.task = self.dict["task"]
        if self.task=="pc":
            self.get_loss = self.get_loss_pc
            self.loss_list = {"pc":0.0, "act":0.0, "weight":0.0}
        elif self.task=="coords":
            self.get_loss = self.get_loss_coords
            self.loss_list = {"coords":0.0, "act":0.0, "weight":0.0}
        elif self.task=="pc_coords":
            self.get_loss = self.get_loss_pc_coords
            self.loss_list = {"pc":0.0, "coords":0.0, "act":0.0, "weight":0.0}
        else:
            print("invalid task:"+str(self.task))
            input()
        
        if init_method=="zero":
            self.prepare_x = self.prepare_x_zero
        elif init_method in ["input", "linear"]:
            self.prepare_x = self.prepare_x_linear
        elif init_method in ["fixed"]:
            self.prepare_x = self.prepare_x_fixed
        elif init_method in ["mlp"]:
            self.prepare_x = self.prepare_x_mlp
        if task in ["pc"]:
            self.prepare_x_0 = self.prepare_x_0_pc
        elif task in ["coords"]:
            self.prepare_x_0 = self.prepare_x_0_coords
        if self.task in ["pc", "pc_coords"]:
            self.dict["pc_error"] = 0.0
        self.loss_count = 0
    def prepare_x_0_pc(self, x_0): #(batch_size, 2)
        return torch.squeeze( self.place_cells.get_activation( torch.unsqueeze( x_0, 1 ) ) ).float()
    def prepare_x_0_coords(self, x_0):
        return x_0.float()
    def prepare_x_linear(self, x, x_0):
        x_0 = self.prepare_x_0(x_0)
        return x, ( torch.unsqueeze( torch.matmul(x_0, self.i_0_h), 0 ), torch.unsqueeze( torch.matmul(x_0, self.i_0_c), 0 ) ) #(batch_size, input_num) x (input_num, N_num) = (batch_size, N_num)
    def prepare_x_fixed(self, x, x_0):
        x_0 = self.prepare_x_0(x_0)
        h_0 = torch.cat([torch.unsqueeze(self.h_0, 0) for _ in range(x.size(0))], dim=0)
        h_0 = torch.unsqueeze(h_0, 0)
        c_0 = torch.cat([torch.unsqueeze(self.c_0, 0) for _ in range(x.size(0))], dim=0)
        c_0 = torch.unsqueeze(c_0, 0)
        return x, (h_0, c_0)
    def prepare_x_mlp(self, x, x_0):
        x_0 = self.prepare_x_0(x_0)
        init_state = torch.unsqueeze( self.encoder(x_0), 0 )
        return x.contiguous(), ( init_state[:, :, 0:self.dict["N_num"]].contiguous(), init_state[:, :, self.dict["N_num"]:(2*self.dict["N_num"])].contiguous() )
    def prepare_x_zero(self, x, x_0):
        return x, None
    def forward(self, x): #(batch_size, sequence_length, input_num)
        x_0 = x[1] #(batch_size, 2:(x_0, y_0))
        x = x[0] #(batch_size, sequence_length, input_num)
        x_0 = x_0.to(device)
        x = x.to(device)
        i_, init_state = self.prepare_x(x, x_0)
        act, (h_n, c_n) = self.lstm(i_, init_state)
        output = torch.matmul(act, self.get_f()) #(batch_size, sequence_length, N_num) x (N_num, output_num) = (batch_size, sequence_length, output_num)
        return output, act
        #return output_list, act_list
    def report_loss(self):
        for key in self.loss_list.keys():
            #print("%s:%s"%(key, str(self.loss_list[key]/self.loss_count)), end=" ")
            print("%s:%.4e"%(key, self.loss_list[key]/self.loss_count), end=" ")
        print("\n")
    def reset_loss(self):
        #print("aaa")
        self.loss_count = 0
        #print(self.loss_count)
        for key in self.loss_list.keys():
            self.loss_list[key] = 0.0
    def get_loss_coords(self, x, y): #x:(batch_size, sequence_length, input_num), y:(batch_size, sequence_length, output_num)
        output, act = self.forward(x)
        self.dict["act_avg"] = torch.mean(torch.abs(act))
        loss_coords = coords_index * F.mse_loss(output, y, reduction='mean')
        loss_act = act_cons_index * torch.mean(act ** 2)
        #loss_weight = torch.zeros((1), device=device)
        loss_weight = weight_cons_index * ( torch.mean(self.f ** 2) )
        self.loss_list["weight"] = self.loss_list["weight"] + loss_weight.item()
        self.loss_list["act"] = self.loss_list["act"] + loss_act.item()
        self.loss_list["coords"] = self.loss_list["coords"] + loss_coords.item()
        self.loss_count += 1
        return loss_coords + loss_act + loss_weight
    def get_loss_pc(self, x, y):
        output, act = self.forward(x)
        self.dict["act_avg"] = torch.mean(torch.abs(act))
        pc_output = self.place_cells.get_activation(y)
        self.dict["pc_error"] = ( torch.sum(torch.abs(output - pc_output)) / torch.sum(torch.abs(pc_output)) ).item() #relative place cells prediction error
        if(self.dict["pc_loss"]=="MSE"):
            loss_pc = index["pc"] * F.mse_loss(output, pc_output, reduction='mean')
        else:
            loss_pc = - torch.mean( pc_output * F.log_softmax(output, dim=2))
            #loss_pc = F.binary_cross_entropy(input=output, target=pc_output)            
            #print(torch.log(pc_output))
            #print(pc_output)
            #print(torch.min(pc_output))
            #print(torch.mean(torch.log(pc_output)))
            #print(loss_pc)
            #input()
        self.dict["pc_avg"] = torch.mean(torch.abs(pc_output))
        #print(output)
        #print(pc_output)
        #print("pc_output")
        #input()
        loss_act = act_cons_index * torch.mean(act ** 2)
        loss_weight = weight_cons_index * ( torch.mean(self.f ** 2) )
        #loss_weight = torch.zeros((1), device=device)
        self.loss_list["weight"] = self.loss_list["weight"] + loss_weight.item()
        self.loss_list["act"] = self.loss_list["act"] + loss_act.item()
        self.loss_list["pc"] = self.loss_list["pc"] + loss_pc.item()
        self.loss_count += 1
        return loss_pc + loss_act + loss_weight
    def get_loss_pc_coords(self, x, y):
        output, act = self.forward(x)
        self.dict["act_avg"] = torch.mean(torch.abs(act))
        pc_output = self.place_cells.get_activation(y)
        loss_coords = index["pc"] * F.mse_loss(output[:, :, 0:2], pc_output, reduction='mean')
        loss_pc = coords_index * F.mse_loss(output[:, :, 2:-1], pc_output, reduction='mean')
        self.dict["pc_error"] = ( torch.sum(torch.abs(output[:, :, 2:-1] - pc_output)) / torch.sum(torch.abs(pc_output)) ).item() #relative place cells prediction error
        loss_act = act_cons_index * torch.mean(act ** 2)
        loss_weight = 0.0
        self.loss_list["weight"] += loss_weight.item()
        self.loss_list["act"] += loss_act.item()
        self.loss_list["coords"] += loss_coords.item()
        self.loss_list["pc"] += loss_pc.item()
        self.loss_count += 1
        return loss_pc + loss_coords + loss_act + loss_weight
    def save(self, net_path):
        f = open(net_path+"state_dict.pth", "wb")
        net = self.to(torch.device("cpu"))
        torch.save(net.dict, f)
        #pickle.dump(net.dict, f)
        net = self.to(device)
        f.close()
    def get_weight(self, name, positive=True, detach=False):
        if name=="r":
            if positive:
                return torch.abs(self.N.get_r_noself())
            else:
                return self.N.get_r()