import random

import numpy as np
from numpy import select, unravel_index

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib as mpl
from matplotlib import pyplot as plt

from utils_torch.attrs import *
import utils_torch
import utils

def InitFromParam(param):
    model = RNNLIF()
    model.InitFromParam(param)
    return model

def load_model(args):
    return 

class RNNLIF(nn.Module):
    # Singel-Layer Recurrent Neural Network with Leaky Integrate-and-Fire Dynamics
    def __init__(self, param=None):
        super(RNNLIF, self).__init__()
        utils_torch.model.InitForModel(self, param)
    def InitFromParam(self, param=None):
        if param is not None:
            self.param = param
        else:
            param = self.param

        param.cache.__object__ = self
        self.data = utils_torch.EmptyPyObj()
        self.cache = utils_torch.EmptyPyObj()
        data = self.data
        cache = self.cache

        utils_torch.AddLog("RNNLIF: Initializing from param...")
        CheckAttrs(param, "Type", value="RNNLIF")
        self.param = param
        #self.json_external_dict = {}
        Neurons = param.Neurons
        EnsureAttrs(param, "Neurons", "IsExciInhi", default=False)
        if MatchAttrs(Neurons.Recurrent, "IsExciInhi", value=True):
            RemoveAttrs(Neurons.Recurrent.IsExciInhi)
            SetAttrs(Neurons.Recurrent.ExciInhi, value={"Enable":True})
        if MatchAttrs(Neurons.Recurrent, "IsExciInhi.Enable", value=True):
            if not HasAttrs(Neurons.Recurrent, "Excitatory.Num"):
                SetAttrs(Neurons.Recurrent, "Excitatory.Num", value=int(Neurons.Recurrent.Num * Neurons.Recurrent.Excitatory.Ratio))
                SetAttrs(Neurons.Recurrent, "Inhibitory.Num", value=(Neurons.Num - Neurons.excitatory.Num))

        cache.Modules = utils_torch.EmptyPyObj()
        cache.Dynamics = utils_torch.EmptyPyObj()

        self.BuildModules()
        self.InitModules()
        self.ParseRouters()
        self.SetForwardEntry()

        EnsureAttrs(param, "InitTasks", default=[])
        for Task in self.param.InitTasks:
            utils_torch.ProcessInitTask(Task, ObjCurrent=self.cache, ObjRoot=utils.ArgsGlobal)
        utils_torch.model.ListParameter(self)
    def Train(self, Input, OutputTarget, TrainParam):
        cache = self.cache
        Output = utils_torch.CallGraph(cache.Dynamics.Train, [Input, OutputTarget, TrainParam])
        return
    def ParseRouters(self):
        cache = self.cache
        param = self.param
        for Name, RouterParam in ListAttrsAndValues(param.Dynamics, Exceptions=["__ResolveRef__", "__Entry__"]):
            utils_torch.router.ParseRouterStatic(RouterParam)
            setattr(cache.Dynamics, Name, RouterParam)
            setattr(RouterParam, "Name", Name) # For Debug
        for Name, RouterParam in ListAttrsAndValues(param.Dynamics, Exceptions=["__ResolveRef__", "__Entry__"]):
            RouterParsed = utils_torch.router.ParseRouterDynamic(RouterParam, InPlace=True, 
                ObjRefList=[cache.Modules, cache.Dynamics, cache, 
                    param.Modules, param.Dynamics, param, self,
                    utils_torch.Models.Operators        
                ])
            setattr(cache.Dynamics, Name, RouterParsed)
        return
    def SetForwardEntry(self):
        param = self.param
        data = self.data
        cache = self.cache
        DefaultEntryStr = "&Dynamics.%s"%ListAttrsAndValues(self.param.Dynamics)[0][0]
        EnsureAttrs(param.Dynamics, "__Entry__", default=DefaultEntryStr)
        SetAttrs(cache, "Dynamics.__Entry__", value=utils_torch.parse.Resolve(param.Dynamics.__Entry__, 
            ObjRefList=[cache.Modules, cache.Dynamics, cache, param.Modules, param.Dynamics, param]))
        #utils_torch.parse.ParseRouters(data.Routers, ObjRefList=[data.Modules, data.Routers, data])
    def forward(self, Input):
        Output = utils_torch.CallGraph(self.cache.Dynamics.__Entry__, Input)
        return Output
    def SetTensorLocation(self, Location):
        utils_torch.model.SetTensorLocationForModel(self, Location)
        self.SetTrainWeight()
    def GetTensorLocation(self):
        return self.cache.TensorLocation
    # def BuildModules(self):
    #     # initialize modules
    #     # for module in ListAttrs(param.modules):
    #     param = self.param
    #     cache = self.cache
    #     for Name, ModuleParam in ListAttrsAndValues(param.Modules):
    #         ModuleParam.Name = Name
    #         ModuleParam.FullName = self.FullName + "." + Name
    #         Module = BuildModule(ModuleParam)
    #         if isinstance(Module, nn.Module):
    #             self.add_module(Name, Module)
    #         setattr(cache.Modules, Name, Module)
    # def InitModules(self):
    #     for name, module in ListAttrsAndValues(self.cache.Modules):
    #         if hasattr(module, "InitFromParam"):
    #             module.InitFromParam()
    #         else:
    #             utils_torch.AddWarning("Module %s has not implemented InitFromParam method."%name)
    def plot_act(self, data=None, ax=None, data_type='u', save=True, save_path='./', save_name='act_map.png', cmap='jet', plot_N_num=200, select_strategy='first', verbose=False):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().Numpy() # [step_num, N_num]

        step_num = data.shape[0]
        N_num = data.shape[1]
        #data = np.transpose(data, (1, 0)) # [N_num, step_num]
        
        if N_num > plot_N_num:
            is_select = True
            if select_strategy in ['first']:
                plot_index = range(plot_N_num)
            elif select_strategy in ['random']:
                plot_index = random.sample(range(N_num), plot_N_num)
            else:
                raise Exception('Invalid select strategy: %s'%select_strategy)
            data = data[:, plot_index]
        else:
            is_select = False
            plot_N_num = N_num

        if ax is None:
            #fig, ax = plt.subplots(figsize = (step_num / 20 * 5, plot_N_num / 20 * 5)) # figsize: (width, height), in inches
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (plot_N_num / 20 * 2, step_num / 20 * 2))
        data_min, data_max, data_mean, data_std = GetItemsFromDict(Getnp_stat(data, verbose=verbose), ['min','max','mean','std'])
        #print(np.argmax(data))
        #print(unravel_index(data.argmax(), data.shape))

        if data_min < data_mean - 3 * data_std:
            data_down = data_mean - 3 * data_std
        else:
            data_down = data_min
        if data_max > data_mean + 3 * data_std:
            data_up = data_mean + 3 * data_std
        else:
            data_up = data_max
        
        if verbose:
            print('data_down:%.3e data_up:%.3e'%(data_down, data_up))
        
        norm = mpl.colors.Normalize(vmin=data_down, vmax=data_up)
        
        data_norm = (data - data_min) / (data_max - data_min)
        
        cmap_func = plt.cm.Getcmap(cmap)
        data_mapped = cmap_func(data_norm)

        im = ax.imshow(data_mapped)
        ax.set_yticks(utils.linspace(0, step_num - 1, 10))
        ax.set_xticks(utils.linspace(0, plot_N_num - 1, 200))
        ax.set_ylabel('Time Step')
        if is_select:
            if select_strategy in ['first']:
                x_label = 'Neuron index'
            elif select_strategy in ['random']:
                x_label = '%d randomly selected Neurons'
            else:
                raise Exception('Invalid select strategy: %s'%select_strategy)
        else:
            x_label = 'Neuron index'
        
        ax.set_xlabel(x_label)

        # plot colorbar
        cbar_ticks = utils.linspace(data_down, data_up, step='auto')
        cbar_ticks_str = list(map(lambda x:'%.2e'%x, cbar_ticks.tolist()))
        cbar = ax.figure.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            ticks=cbar_ticks,
            ax=ax)
        cbar.ax.set_yticklabels(cbar_ticks_str)  # vertical colorbar. use set_xticklabels for horizontal colorbar

        if data_type in ['u']:
            ax.set_title('Firing rate across time')
            cbar.set_label('Firing rate')
        else:
            ax.set_title('Membrane potential rate across time')
            cbar.set_label('Membrane potential')
        
        if save:
            EnsurePath(save_path)
            plt.savefig(save_path + save_name)
        return ax
    '''
    def cal_perform_coord(self, data):
        output_truth = GetItemsFromDict(data, ['output'])
        # x: [batch_size, step_num, input_num]
        # y: [batch_size, step_num, output_num]
        output, act = GetItemsFromDict(self.forward(data), ['output', 'act'])
        self.dict['act_avg'] = torch.mean(torch.abs(act))
        loss_coords = self.main_coeff * F.mse_loss(output, y, reduction='mean')
        #loss_coords = 0.0
        #for time in range(x.size(1)):
        #    loss_coords = loss_coords + F.mse_loss(output[time], torch.squeeze(y[:, time, :]), reduction='mean')
        #loss_coords = coords_index * loss_coords / (x.size(1))
        loss_act = self.act_coeff * torch.mean(act ** 2)
        #loss_act = 0.0
        #for time in range(x.size(1)):
        #    loss_act = loss_act + torch.mean(act[time] ** 2)
        #loss_act = loss_act / (x.size(1))
        loss_weight = self.weight_coeff * ( torch.mean(self.Geto() ** 2) + torch.mean(self.Geti() ** 2) )
        self.perform_list['weight'] += loss_weight.item()
        self.perform_list['act'] += loss_act.item()
        self.perform_list['coord'] += + loss_coords.item()
        self.batch_count += 1
        #self.sample_count += self
        return loss_coords + loss_act + loss_weight
    def cal_perform_pc(self, data):
        x, x_init, output_truth = GetItemsFromDict(data, ['input', 'input_init', 'output'])
        batch_size = x.size(0)
        output, act = GetItemsFromDict(self.forward(data), ['output', 'act'])
        
        self.dict['act_avg'] = torch.mean(torch.abs(act))
        
        if self.main_loss in ['mse', 'MSE']:
            loss_main = self.main_coeff * F.mse_loss(output, output_truth, reduction='mean')
            self.perform_list['output_error_ratio'] += ( torch.sum(torch.abs(output - output_truth)) / torch.sum(torch.abs(output_truth)) ).item() # relative place cells prediction error
        elif self.main_loss in ['cel', 'CEL']:
            loss_main = - self.main_coeff * torch.mean( output_truth * F.log_softmax(output, dim=2) )
        else:
            raise Exception('Invalid main loss: %s'%str(self.main_loss))
        
        self.dict['pc_avg'] = torch.mean(torch.abs(output_truth))
        loss_act = self.act_coeff * torch.mean(act ** 2)
        
        loss_weight_0 = self.cal_loss_weight()
        #loss_weight_0 = self.weight_coeff * torch.mean(self.Getr() ** 2)
        #print(loss_weight_0)
        # dynamically alternate weight Coefficient
        #print(loss_weight_0.size())
        if self.weight_coeff > 0.0:
            if self.dynamic_weight_coeff:
                loss_ratio = loss_weight_0.item() / loss_main.item() # ratio of weight loss to pc loss.
                #print('loss_ratio:%.3f'%loss_ratio)
                if self.weight_ratio_min < loss_ratio < self.weight_ratio_max:
                    loss_weight = loss_weight_0
                else:
                    weight_coeff_0 = self.weight_coeff
                    self.weight_coeff = self.weight_coeff * self.weight_ratio / loss_ratio # alternating weight cons index so that loss_weight == loss_main * dynamic_weight_ratio
                    self.alt_weight_coeff_count += 1
                    if self.alt_weight_coeff_count > 50:
                        print('alternating weight_coeff from %.3e to %.3e'%(weight_coeff_0, self.weight_coeff))  
                        self.alt_weight_coeff_count = 0
                    loss_weight = self.weight_coeff / weight_coeff_0 * loss_weight_0
            else:
                loss_weight = loss_weight_0
        
        self.perform_list['weight'] += loss_weight.item()
        self.perform_list['act'] += loss_act.item()
        self.perform_list['main'] += loss_main.item()
        
        self.batch_count += 1
        self.sample_count += batch_size
        return {
            'loss_main': loss_main,
            'loss_act': loss_act,
            'loss_weight': loss_weight,
            'loss': loss_main + loss_act + loss_weight
        }
    def cal_perform_pc_coord(self, data):
        #x, y = self.prep_path(path)
        y = GetItemsFromDict(data, ['output'])
        batch_size = y.size(0)
        output, act = GetItemsFromDict(self.forward(data), ['output', 'act'])
        self.dict['act_avg'] = torch.mean(torch.abs(act))
        pc_output = self.place_cells.Getact(y)
        loss_coords =self.main_coeff_pc * F.mse_loss(output[:, :, 0:2], pc_output, reduction='mean')
        loss_main = self.main_coeff_coords * F.mse_loss(output[:, :, 2:-1], pc_output, reduction='mean')
        self.perform_list['pc_error_ratio'] += ( torch.sum(torch.abs(output[:, :, 2:-1] - pc_output)) / torch.sum(torch.abs(pc_output)) ).item() #relative place cells prediction error
        
        loss_act = self.act_coeff * torch.mean(act ** 2)
        
        loss_weight = self.weight_coeff * ( torch.mean(self.Geto() ** 2) + torch.mean(self.Geti() ** 2) )
        
        self.perform_list['weight'] += loss_weight.item()
        self.perform_list['act'] += loss_act.item()
        self.perform_list['coord'] += loss_coords.item()
        self.perform_list['pc'] += loss_main.item()
        
        self.batch_count += 1
        self.sample_count += batch_size
        return {
            'loss_main': loss_main,
            'loss_coord': loss_coords,
            'loss_act': loss_act,
            'loss_weight': loss_weight,
            'loss': loss_main + loss_coords + loss_act + loss_weight
        }
    '''
    def cal_loss_weight_(self, coeff):
        #if weight_cons is None:
        #    weight_cons = self.weight_cons
        weight_cons = self.weight_cons
        #if coeff == 0.0:
        #    return torch.tensor([0.0], device=self.device)
        loss = 0.0
        for Getweight in weight_cons:
            weight = Getweight()
            if isinstance(weight, torch.Tensor):
                loss += torch.mean(weight ** 2)
        loss = coeff * loss
        return loss
    def alt_pc_act_strength(self, path, verbose=True):
        pc_mean, pc_pred_mean = self.Getoutput_ratio_pc(path, verbose)
        act_center_0 = self.place_cells.act_center
        self.place_cells.act_center = act_center_1 = 1.0 * act_center_0 * pc_pred_mean / pc_mean
        if verbose:
            print('alternating pc peak activation from %.3e to %.3e'%(act_center_0, act_center_1))
    def Getoutput_ratio_pc(self, path, verbose):
        x, y = self.prep_path(path)
        pc_output = self.place_cells.Getact(y)
        output, act = GetItemsFromDict(self.forward(x), ['output', 'act'])
        pc_mean = torch.mean(pc_output).item()
        pc_pred_mean = torch.mean(output).item()
        if verbose:
            print('pc_act mean: %.3e pc_act_pred_mean: %.3e'%(pc_mean, pc_pred_mean))
        return pc_mean, pc_pred_mean
    def Getoutput_from_act(self, act, to_array=True):
        if isinstance(act, np.ndarray):
            # isinstance(act, type(np.ndarray)) does not work.
            act = torch.from_numpy(act).to(self.device).float()
        #print(act.size())
        output = torch.mm(act, self.Geto())
        if to_array:
            output = output.detach().cpu().Numpy()
        return output

    def save(self, save_path, save_name):
        EnsurePath(save_path)
        #self.update_before_save()
        with open(save_path + save_name, 'wb') as f:
            self.to(torch.device('cpu'))
            torch.save(self.dict, f)
            self.to(self.device)
    def Getweight(self, name, detach=True):
        if name in ['r']:
            w = self.Getr()
        elif name in ['o']:
            w = self.Geto()
        else:
            raise Exception('Invalid weight name: %s'%name)
        if detach:
            w = w.detach()
        return w
    def anal_weight_change_(self):
        for name, value in self.named_parameters():
            #if name in ['encoder.0.weight','encoder.2.weight']:
            if True:
                #print('name: {0},\t grad: {1}'.format(name, value.requires_grad))
                #print(value)
                #print(value.grad)
                value_np = value.detach().cpu().Numpy()
                if self.cache.get(name) is not None:  
                    #print('  change in %s: '%name, end='')
                    #print(value_np - self.cache[name])
                    print('  ratio change in %s: '%name, end='')
                    print( np.sum(np.abs(value_np-self.cache[name])) / np.sum(np.abs(self.cache[name])) )
                self.cache[name] = value_np
    def anal_weight_change(self, verbose=True):
        result = ''
        r_1 = self.Getr().detach().cpu().Numpy()
        if self.cache.get('r') is not None:
            r_0 = self.cache['r']
            r_change_rate = np.sum(abs(r_1 - r_0)) / np.sum(np.abs(r_0))
            result += 'r_change_rate: %.3e '%r_change_rate
        self.cache['r'] = r_1

        o_1 = self.Geto().detach().cpu().Numpy()
        if self.cache.get('o') is not None:
            o_0 = self.cache['o']
            f_change_rate = np.sum(abs(o_1 - o_0)) / np.sum(np.abs(o_0))
            result += 'f_change_rate: %.3e '%f_change_rate
        self.cache['o'] = o_1

        if hasattr(self, 'Geti'):
            i_1 = self.Geti().detach().cpu().Numpy()
            if self.cache.get('i') is not None:
                i_0 = self.cache['i']
                i_change_rate = np.sum(abs(i_1 - i_0)) / np.sum(np.abs(i_0))
                result += 'i_change_rate: %.3e '%i_change_rate
            self.cache['i'] = i_1
        if verbose:
            print(result)
        return result
    def Getweight_stat(self, verbose=True, complete=False):
        result = ''
        for name in ['i', 'r', 'o']:
            if hasattr(self, name):
                result += Gettensor_stat(getattr(self, name), name=name, verbose=False, complete=complete)
        if verbose:
            print(result)
        return result
    def Getweight_info(self, verbose=True, complete=False):
        result = ''
        for name in ['i', 'r', 'o']:
            if hasattr(self, name):
                result += Gettensor_info(getattr(self, name), name=name, verbose=False, complete=complete)
        if verbose:
            print(result)
        return result
    def anal_gradient(self, verbose=True):
        result = ''
        for name in ['i', 'r', 'o']:
            if hasattr(self, name):
                weight = getattr(self, name)
                if weight.grad is not None:
                    ratio = torch.sum(torch.abs(weight.grad)) / torch.sum(torch.abs(weight))
                    result += '%s: ratio_grad_weight: %.3e ' % (name, ratio)
        if verbose:
            print(result)
        return result
    '''
    def prep_path(self, path):
        if self.input_mode in ['v_xy']:
            inputs = torch.from_numpy(path['xy_delta']).float() # [batch_size, step_num, (vx, vy)]
            outputs = torch.from_numpy(path['xy']).float() # [batch_size, step_num, (x, y)]
        elif self.input_mode in ['v_hd']:
            inputs = torch.from_numpy(np.stack((path['theta_xy'][:,:,0], path['theta_xy'][:,:,1], path['delta_xy']), axis=-1)).float() # [batch_size, step_num, (cos, sin, v)]
            outputs = torch.from_numpy(path['xy']).float() # [batch_size, step_num, (x, y)]
        else:
            raise Exception('Unknown input mode:'+str(self.input_mode))
        init = torch.from_numpy(path['xy_init']).float() # [batch_size, 2]
        inputs = inputs.to(self.device)
        init = init.to(self.device)
        outputs = outputs.to(self.device)
        return (inputs, init), outputs
    '''
    def plot_Recurrent_weight(self, ax, cmap):
        weight_r = self.Getr().detach().cpu().Numpy()
        weight_r_mapped, weight_min, weight_max = norm_and_map(weight_r, cmap=cmap, return_min_max=True) # weight_r_mapped: [N_num, res_x, res_y, (r,g,b,a)]
        
        ax.set_title('Recurrent weight')
        ax.imshow(weight_r_mapped, extent=(0, self.N_num, 0, self.N_num))

        norm = mpl.colors.Normalize(vmin=weight_min, vmax=weight_max)
        ax_ = ax.inset_axes([1.05, 0.0, 0.12, 0.8]) # left, bottom, width, height. all are ratios to sub-canvas of ax.
        cbar = ax.figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), 
            cax=ax_, # occupy ax_ 
            ticks=np.linspace(weight_min, weight_max, num=5),
            orientation='vertical')
        cbar.set_label('Connection strength', loc='center')
        
        if self.separate_ei:
            #ax.set_xticklabels('')
            #ax.set_yticklabels('')
            ax.set_xticks([0, self.E_num, self.N_num])
            ax.set_yticks([0, self.E_num, self.N_num])

            ax.set_xticks([(0 + self.E_num)/2, (self.E_num + self.N_num)/2], minor=True)
            ax.set_xticklabels(['to E', 'to I'], minor=True)

            ax.set_yticks([(0 + self.E_num)/2, (self.E_num + self.N_num)/2], minor=True)
            ax.set_yticklabels(['from E', 'from I'], minor=True)
            
            ax.tick_param(axis='both', which='minor', length=0)

        else:
            ax.set_xticks([0, self.N_num])
            ax.set_yticks([0, self.N_num])            

        ax.set_xlabel('Postsynaptic neuron index')
        ax.set_ylabel('Presynaptic neuron index')
    
    def plot_weight(self, ax=None, save=True, save_path='./', save_name='RNN_Navi_weight_plot.png', cmap='jet'):
        if ax is None:
            plt.close('all')
            row_num, col_num = 2, 2
            fig, axes = plt.subplots(nrows=row_num, ncols=col_num, figsize=(5*col_num, 5*row_num)) # figsize unit: inches

        fig.suptitle('Weight Visualization of 1-layer RNN')

        # plot Recurrent weight
        ax = axes[0, 0] # raises error is row_num==col_num==1
        
        self.plot_Recurrent_weight(ax, cmap)

        # plot input_weight
        if self.init_method in ['linear']:
            ax = axes[0, 1]
        elif self.init_method in ['mlp']:
            pass
        else:
            pass
        plt.tight_layout()
        if save:
            utils_torch.EnsureFileDir(save_path)
            plt.savefig(save_path + save_name)
    def GetTrainWeight(self):
        return self.cache.TrainWeight
    def SetTrainWeight(self):
        return utils_torch.model.SetTrainWeightForModel(self)
    def ClearTrainWeight(self):
        utils_torch.model.ClearTrainWeightForModel(self)
    # def SetLogger(self, logger):
    #     return utils_torch.model.SetLoggerForModel(self, logger)
    # def GetLogger(self):
    #     return utils_torch.model.GetLoggerForModel(self)
    # def Log(self, data, Name="Undefined"):
    #     return utils_torch.model.LogForModel(self, data, Name)
    def SetFullName(self, FullName):
        utils_torch.model.SetFullNameForModel(self, FullName)

__MainClass__ = RNNLIF
utils_torch.model.SetMethodForModelClass(__MainClass__)