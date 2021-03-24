import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
#training parameters.

import numpy as np
import math
import cmath
import random

import copy

def save_dict(net, dict=None):
    return net.dict
def load_dict(net, f):
    net.dict=pickle.load(f)

def get_weight(net, name=""):
    try:
        if name in [""]:
            return net.weight_names
        else:
            names = name.split(".",1)
            return net.name_index[(names[0].split("_",1))[0]].get_weight(names[1])
    except Exception:
        return "error"

def set_constraint_names(net):
    net.weight_names=[]
    net.constraint_weight_names=[]
    for name in net.neurons:
        net.weight_names.append(name+".f")
        net.weight_names.append(name+".l")
        net.weight_names.append(name+".r")
        if(net.f_Dale==True):
            net.constraint_names.append(name+".f")
        if(net.rl_Dale==True):
            net.constraint_names.append(name+".l")
            net.constraint_names.append(name+".r")

def get_weight_index(init_des=""):
    index = 1.00e-03
    ex_large_index = 1.00
    large_index = 0.10
    middle_index = 0.01
    if init_des in ["ex-large"]:
        index=ex_large_index
    if init_des in ["large"]:
        index=large_index
    if init_des in ["middle"]:
        index=middle_index
    return index

def get_ei_mask(E_num, N_num, device=None):
    ei_mask = torch.zeros((N_num, N_num), device=device, requires_grad=False)
    for i in range(E_num):
        ei_mask[i][i] = 1.0
    for i in range(E_num, N_num):
        ei_mask[i][i] = -1.0
    return ei_mask

def get_mask(N_num, output_num, device=None):
    mask = torch.ones((N_num, output_num), device=device, requires_grad=False)
    return mask

def set_act_func(net, act_func_des="relu"):
    if(isinstance(act_func_des, str)):
        set_act_func_from_name(net, act_func_des)
    elif(isinstance(act_func_des, dict)):
        if(net.dict.get("type")!=None and act_func_des.get(net.type)!=None):
            set_act_func_from_name(net, act_func_des[net.type])
        else:
            set_act_func_from_name(net, act_func_des["default"])
def get_act_func_module(act_func_des):
    name=act_func_des
    if name=="relu":
        return nn.ReLU()
    elif name=="tanh":
        return nn.Tanh()
    elif name=="softplus":
        return nn.Softplus()
    elif name=="sigmoid":
        return nn.Sigmoid()

def get_act_func_from_name(name="relu", param="default"):
    if(name=="none"):
        return lambda x:x
    elif(name=="relu"):
        if(param=="default"):
            return lambda x:F.relu(x)
        else:
            return lambda x:param * F.relu(x)
    elif(name=="tanh"):
        if(param=="default"):
            return lambda x:torch.tanh(x)
        else:
            return lambda x:param * F.tanh(x)
    elif(name=="relu_tanh"):
        if(param=="default"):
            return lambda x:F.relu(torch.tanh(x))
        else:
            return lambda x:param * F.relu(torch.tanh(x))

def get_act_func(act_func_des):
    if(isinstance(act_func_des, list)):
        act_func_name = act_func_des[0]
        act_func_param = act_func_des[1]
    elif(isinstance(act_func_des, str)):
        act_func_name = act_func_des
        act_func_param = "default"
    elif(isinstance(act_func_des, dict)):
        act_func_name = act_func_des["name"]
        act_func_param = act_func_des["param"]
    return get_act_func_from_name(act_func_name, act_func_param)

def init_weight(weight, params, weight_name="unnamed"):
    dim_num = len(list(weight.size()))
    name = get_name(params)
    print("weight:%s init_method:%s"%(weight_name, str(name)))
    coeff = get_arg(params)
    sig = False
    if dim_num==1:
        divider = weight.size(0)
        sig=True
    elif dim_num==2:
        if name=="output":
            divider = weight.size(1)
            sig=True
        elif name=="input":
            divider = weight.size(0)
            sig=True
        elif name in ["ortho", "orthogonal"]:
            weight_ = weight.detach().clone()
            torch.nn.init.orthogonal_(weight_, gain=1.0) #init input weight to be orthogonal.
            with torch.no_grad():  #avoid gradient calculation error during in-place operation.
                weight.copy_( weight_ * coeff )
            return
        elif name in ["glorot", "glorot_uniform", "xavier", "xavier_uniform"]:
            weight_ = weight.detach().clone()
            torch.nn.init.xavier_uniform_(weight_, gain=1.0)
            with torch.no_grad():
                weight.copy_( weight_ * coeff )
            return
    if sig:
        lim = coeff / divider
        if param_config.constraint_method=="force":
            torch.nn.init.uniform_(weight, 0.0, lim)
        else:
            torch.nn.init.uniform_(weight, -lim, lim)



def load_mlp(dict_):
    act_func = get_act_func_module(dict_["act_func"])
    layers = []
    N_nums = dict_["N_nums"] #input_num, hidden_layer1_unit_num, hidden_layer2_unit_numm ... output_num
    layer_num = len(N_nums) - 1
    for layer_index in range(layer_num):
        print(layer_index)
        current_layer = nn.Linear(N_nums[layer_index], N_nums[layer_index+1], bias=dict_["bias"])
        current_layer.load_state_dict(dict_["layer_dicts"][layer_index])
        layers.append(current_layer)
        if not (dict_["act_func_on_last_layer"] and layer_index==layer_num-1):
            layers.append(act_func)
    return torch.nn.Sequential(*layers)
def update_mlp(dict_, layers):
    layer_num = len(layers)
    count = 0
    for layer_index in range(layer_num):
        if isinstance(layers[layer_index], nn.Linear):
            dict_["layer_dicts"][count] = layers[layer_index].state_dict()
            count += 1
        else:
            pass
def build_mlp(dict_):
    act_func = get_act_func_module(dict_["act_func"])
    layers = []
    layer_dicts = []
    N_nums = dict_["N_nums"] #input_num, hidden_layer1_unit_num, hidden_layer2_unit_numm ... output_num
    layer_num = len(N_nums) - 1
    for layer_index in range(layer_num):
        current_layer = nn.Linear(N_nums[layer_index], N_nums[layer_index+1], bias=dict_["bias"])
        layers.append(current_layer)
        layer_dicts.append(current_layer.state_dict())
        if not (dict_["act_func_on_last_layer"] and layer_index==layer_num-1):
            layers.append(act_func)
    dict_["layer_dicts"] = layer_dicts
    return torch.nn.Sequential(*layers), dict_, layers

def to_np_array(data):
    if(isinstance(data, list)):
        data=np.array(data)
    if(isinstance(data, torch.Tensor)):
        try:
            data=data.numpy()
        except Exception:
            data=data.detach().cpu().numpy()
    return data