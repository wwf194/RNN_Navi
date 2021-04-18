import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib as mpl
from matplotlib import pyplot as plt

from utils import set_instance_variable, contain, get_name, get_items_from_dict, get_name_args, ensure_path
#from utils_anal import *
from utils_model import get_act_func, get_mask, get_ei_mask, init_weight, build_mlp, print_model_param
from utils_plot import norm_and_map
#from Place_Cells import Place_Cells
#from Neurons_LIF import Neurons_LIF

class RSLP_LIF(nn.Module): # recurrent single layer perceptron with leak-integrate-and-fire cell state
    def __init__(self, dict_=None, load=False):
        super(RSLP_LIF, self).__init__()
        self.dict = dict_
        
        #set_instance_variable(self, self.dict)
        self.separate_ei = self.dict['separate_ei']
        self.load = load
        self.device = self.dict.setdefault('device', 'cpu')
        self.N_num = self.dict['N_num']
        self.input_num = self.dict['input_num']
        self.input_init_num = self.dict['input_init_num']
        self.output_num = self.dict['output_num']
        if self.dict.get('time_const_e') is not None and self.dict.get('time_const_i') is not None:
            self.time_const_e = self.dict['time_const_e']
            self.time_const_i = self.dict['time_const_i']
        else:
            self.time_const = self.dict['time_const']
        if self.separate_ei:
            self.E_num = self.dict['E_num']
            self.I_num = self.dict['I_num']
            self.weight_Dale = self.dict['weight_Dale']
            #set_instance_variable(self, self.dict, keys=['E_num', 'I_num', 'weight_Dale'])

        # set up weights and biases
        if load:
            '''
            self.i = self.dict['i'] # input weight
            self.r = self.dict['r']
            self.r_b = self.dict['r_b']
            self.o = self.dict['o']
            '''
            pass
        else:
            self.dict.setdefault('r_bias', True)
            if self.dict['r_bias']:
                self.r_b = self.dict['r_b'] = nn.Parameter(torch.zeros((self.N_num), device=self.device, requires_grad=True))
            else:
                self.r_b = self.dict['r_b'] = 0.0
            #self.i = self.dict['i'] = nn.Parameter(torch.zeros((self.input_num, self.N_num), device=self.device, requires_grad=True)) #input weights
            self.o = self.dict['o'] = nn.Parameter(torch.zeros((self.N_num, self.output_num), device=self.device, requires_grad=True))
            self.r = self.dict['r'] = nn.Parameter(torch.zeros((self.N_num, self.N_num), device=self.device, requires_grad=True))
            if self.dict.get('init_weight') is None:
                self.dict['init_weight'] = {
                    #'i': ['input', 1.0],
                    'r': ['input', 1.0],
                    'o': ['input', 1.0],
                }
            else:
                self.init_weight = self.dict['init_weight']
            #init_weight(self.dict['i'], self.dict['init_weight']['i'])
            init_weight(self.dict['r'], self.dict['init_weight']['r'])
            init_weight(self.dict['o'], self.dict['init_weight']['o'])
        #self.register_parameter('i', self.i)
        self.register_parameter('r', self.dict['r'])
        if isinstance(self.dict['r_b'], torch.Tensor):
            self.register_parameter('r_b', self.dict['r_b'])
        else:
            self.r_b = self.dict['r_b']
        self.register_parameter('o', self.dict['o'])
    
        # set up init method
        # for compatibility with older versions
        '''
        if self.input_mode not in ['v_xy', 'v_hd']: 
            self.input_mode = 'v_xy'
        '''
        if self.dict.get('init_method') is None:
            self.dict['init_method'] = self.dict['init_mode']
        self.init_method_name, self.init_method_dict = get_name_args(self.dict['init_method'])
        #print('init_method: %s'%self.init_method)
        if self.init_method_name in ['zero']:
            self.cal_init_state = self.cal_init_state_zero
            self.reset = self.reset_zero
        elif self.init_method_name in ['fixed', 'learnable']: # init with fixed and learnable state.
            if load:
                self.s_init = self.dict['s_init']
                self.h_init = self.dict['h_init']
                # ? necessary?
                self.s_init.requires_grad = True
                self.h_init.requires_grad = True

            else:
                self.s_init = self.dict['s_init'] = torch.zeros((self.dict['N_num']), device=self.device, requires_grad=True)
                self.h_init = self.dict['h_init'] = torch.zeros((self.dict['N_num']), device=self.device, requires_grad=True)
                init_weight(self.h_init, self.dict['init_weight']['s_init'])
                init_weight(self.h_init, self.dict['init_weight']['h_init'])    
            self.register_parameter(self.s_init, 's_init')
            self.register_parameter(self.h_init, 'h_init')
            self.reset = self.reset_fixed
        elif self.init_method_name in ['linear']:
            if self.init_method_dict.get('N_nums') is None:
                self.init_method_dict['N_nums'] = [self.input_init_num, self.N_num * 2]
            '''
            dict_={
                'type':'linear',
                'N_nums': [self.input_init_num, self.N_num * 2],
                'act_func_on_last_layer': False,
                'bias': True,
                'bias_on_last_layer': True,
                'batch_norm': True,
            }
            '''
            self.mlp_init = build_mlp(
                dict_=self.init_method_dict,
                device=self.device,
                load=self.load
            )
            self.cal_init_state = self.cal_init_state_mlp
            self.add_module('mlp_init', self.mlp_init)
        elif self.init_method_name in ['given']:
            self.cal_init_state = self.cal_init_state_given
        elif self.init_method_name in ['mlp', 'MLP']:
            if self.init_method_dict.get('N_nums') is None:
                layer_num = self.init_method_dict['layer_num']
                N_nums = [self.input_init_num]
                for _ in range(layer_num):
                    N_nums.append(self.N_num * 2)
                self.init_method_dict['N_nums'] = N_nums
            self.mlp_init = build_mlp(
                dict_=self.init_method_dict,
                device=self.device,
                load=self.load
            )
            self.cal_init_state = self.cal_init_state_mlp
            self.add_module('mlp_init', self.mlp_init)
        else:
            raise Exception('Invalid init method: %s'%(self.init_method_name))

        # set up input method
        self.input_method_name, self.input_method_dict = get_name_args(self.dict['input_method'])
        if self.input_method_name in ['linear']:
            if self.input_method_dict.get('N_nums') is None:
                self.input_method_dict['N_nums'] = [self.input_num, self.N_num]
            self.mlp_input = build_mlp(
                dict_=self.input_method_dict,
                device=self.device,
                load=self.load
            )
            self.add_module('mlp_input', self.mlp_input)
            self.cal_input = self.cal_input_mlp
            self.i = self.mlp_input.params['w0']
            self.i_b = self.mlp_input.params['b0']
            self.get_i = lambda:self.i
        elif self.input_method_name in ['mlp']:
            if self.input_method_dict.get('N_nums') is None:
                layer_num = self.input_method_dict['layer_num']
                N_nums = [self.input_init_num]
                for _ in range(layer_num):
                    N_nums.append(self.N_num * 2)
                self.input_method_dict['N_nums'] = N_nums
            self.mlp_input = build_mlp(
                dict_=self.input_method_dict,
                device=self.device,
                load=self.load
            )
            self.add_module('mlp_input', self.mlp_input)
            self.cal_input = self.cal_input_mlp
        else:
            raise Exception('Invalid input method: %s'%self.input_method_name)

        # set up recurrent weight
        if self.dict['no_self']:
            self.r_self_mask = torch.ones( (self.N_num, self.N_num), device=self.device, requires_grad=False )
            for i in range(self.dict['N_num']):
                self.r_self_mask[i][i] = 0.0
            self.get_r_noself = lambda :self.r * self.r_self_mask
        else:
            self.get_r_noself = lambda :self.r
        self.ei_mask = None
        if self.dict['separate_ei'] and 'r' in self.weight_Dale:
            self.ei_mask = get_ei_mask(E_num=self.dict['E_num'], N_num=self.dict['N_num'], device=self.device)
            self.get_r_ei = lambda :torch.mm(self.ei_mask, torch.abs(self.get_r_noself()))
        else:
            self.get_r_ei = self.get_r_noself

        disable_connection = self.dict.setdefault('disable_connection', [])
        print('disable_connection:'+str(disable_connection))
    
        if len(disable_connection)>0:
            self.mask = torch.ones( (self.N_num, self.N_num), device=self.device, requires_grad=False )
            if 'E->E' in disable_connection:
                #print('E->E connection disabled.')
                self.mask[0:self.E_num, 0:self.E_num] = 0.0
            if 'E->I' in disable_connection:
                #print('bannning E->I connection disabled.')
                self.mask[0:self.E_num, self.E_num:self.N_num] = 0.0
            if 'I->E' in disable_connection:
                #print('bannning I->E connection disabled.')
                self.mask[self.E_num:self.N_num, 0:self.E_num] = 0.0
            if 'I->I' in disable_connection:
                #print('bannning I->I connection disabled.')
                self.mask[self.E_num:self.N_num, self.E_num:self.N_num] = 0.0
            self.get_r_mask = lambda :self.mask * self.get_r_ei()
        else:
            self.get_r_mask = self.get_r_ei
        
        self.get_r = self.get_r_mask

        # set up output weight
        if self.separate_ei and 'o' in self.dict['weight_Dale']: #set mask for EI separation
            if self.ei_mask is None:
                self.ei_mask = get_ei_mask(E_num=self.dict['E_num'], N_num=self.dict['N_num'], device=self.device)
            self.get_o_ei = lambda :torch.mm(self.ei_mask, torch.abs(self.o))
        else:
            self.get_o_ei = lambda :self.o
        self.get_o = self.get_o_ei
 
        # set up noise
        self.noise_coeff = self.dict.setdefault('noise_coeff', 0.0)
        if self.noise_coeff==0.0:
            self.get_noise = lambda batch_size:0.0
        else:
            self.get_noise = self.get_noise_gaussian
        
        # set up act_func
        self.act_func = get_act_func(self.dict['act_func'])
        '''
        self.drop_out = self.dict.setdefault('drop_out', False)
        if self.drop_out:
            self.drop_out = torch.nn.Dropout(p=0.5)
        '''

        '''
        if self.task in ['pc', 'pc_coord', 'coords_pc']:
            #self.place_cells = Place_Cells(dict_ = self.dict['place_cells'], load=self.load)
            #self.place_cells.receive_options(options)
            self.pc_num = self.dict['place_cells']['N_num']
            
        if self.task in ['pc']:
            self.get_perform = self.get_loss = self.get_perform_pc
            self.perform_list = {'pc':0.0, 'act':0.0, 'weight':0.0}
            self.dict_N['i0_size'] = self.pc_num
            self.prep_x0 = self.prep_x0_pc
            self.output_mode = self.dict['output_mode'] = 'pc'
        elif self.task in ['pc_coord', 'coords_pc']:
            #self.place_cells = Place_Cells(dict_ = self.dict['place_cells'], load=self.load)
            self.get_perform = self.get_loss = self.get_perform_pc_coords
            self.perform_list = {'pc':0.0, 'coord':0.0, 'act':0.0, 'weight':0.0}
            self.dict_N['i0_size'] = self.pc_num + self.input_num
            self.output_mode = self.dict['output_mode'] = 'pc_coord'
        elif self.task in ['coord']:
            self.get_perform = self.get_loss = self.get_perform_coords
            self.perform_list = {'coord':0.0, 'act':0.0, 'weight':0.0}
            self.dict_N['i0_size'] = self.input_num
            self.prep_x0 = self.prep_x0_null
            self.output_mode = self.dict['output_mode'] = 'coord'
        else:
            raise Exception('RNN_Navi: Invalid task: %s'%str(self.task))
        '''
        
        if self.dict.get('weight_cons_name') is None:
            self.dict['weight_cons_name'] = ['r']
        self.weight_cons = []
        self.cal_loss_weight = self.cal_loss_weight_
        for weight_name in self.dict['weight_cons_name']:
            if weight_name in ['r']:
                self.weight_cons.append(self.get_r)
            elif weight_name in ['o']:
                self.weight_cons.append(self.get_o)
            else:
                raise Exception('Invalid weight name: %s'%weight_name)
        
        self.cache = {}
    def cal_init_state_zero(self, **kw):
        batch_size = kw['batch_size']
        s_h_init = {
            's_init': torch.zeros([batch_size, self.N_num], device=self.device), # [batch_size, N_num]
            'h_init': torch.zeros([batch_size, self.N_NUM], device=self.device), # [batch_size, N_num]
        }
        return s_h_init
    def cal_init_state_fixed(self, **kw):
        batch_size = kw['batch_size']
        s_h_init = {
            's_init': torch.stack([self.s_init for _ in range(batch_size)], dim=0),
            'h_init': torch.stack([self.h_init for _ in range(batch_size)], dim=0),
        }
        return s_h_init
    def cal_init_state_mlp(self, **kw):
        x_init = kw['x_init'].to(self.device)
        mlp_output = self.mlp_init(x_init) #[batch_size, 2 * N_num]
        s_h_init = {
             's_init': mlp_output[:, 0:self.N_num],
             'h_init': mlp_output[:, self.N_num:]
        }
        return s_h_init
    def cal_init_state_given(self, **kw):
        s_h_init = {
            's_init': kw['s_init'],
            'h_init': kw['h_init'],
        }
        return s_h_init
    def cal_input_batch_time(self, **kw):
        x = kw['x'] # [batch_size, step_num, input_num]
        batch_size, step_num, input_num = x.size(0), x.size(1), x.size(2)
        x = x.view(batch_size * step_num, input_num)
        i = self.cal_input(x=x) # [batch_size * step_num, N_num]
        i = i.view(batch_size, step_num, self.N_num)
        return i
    def cal_input_mlp(self, **kw):
        x = kw['x'] # [batch_size, input_num]
        i = self.mlp_input(x)
        return i
    
    '''
    def prep_x0_pc(self, x0): # [batch_size, input_num]
        #return torch.squeeze(self.place_cells.get_act(torch.unsqueeze(x0, 1))) # [batch_size, pc_num]
        # to be implemented: batch_size must be > 1.
        return torch.squeeze(self.place_cells.get_act(torch.unsqueeze(x0, 1))).float() # [batch_size, pc_num]
    def prep_x0_null(self, x0):
        return x0 # [batch_size, input_num]
    '''
    #def receive_options(self, options):

    '''
    def update_before_save(self):
        if get_name(self.dict['init_method']) in ['mlp']:
            update_mlp(self.dict['encoder_dict'], self.encoder_layers)
    '''
    def forward(self, data): # ([batch_size, step_num, input_num], [batch_size, input_num])
        x, x_init = get_items_from_dict(data, ['input', 'input_init'])
        x_init = x_init.to(self.device) # [batch_size, input_init_num]
        x = x.to(self.device) # [batch_size, step_num, input_num]

        i = self.cal_input_batch_time(x=x) # [batch_size, step_num, N_num]

        x_size = list(x.size())
        #i_ = x.mm(self.get_i()) #[batch_size, step_num, N_num]
        '''
        x = x.view(x_size[0] * x_size[1], x_size[2])
        i_ = ( x.mm(self.get_i()) ) # [batch_size, step_num, N_num]
        i_ = i_.view(x_size[0], x_size[1], -1)
        r0 = self.reset(x=x, i0=self.prep_x0(x_init))
        '''
        '''
        if pre_input:
            x0 = torch.squeeze(x[:, 0, :]).detach().cpu().numpy() #(batch_size, input_num)
            x0 = torch.from_numpy(x0).to(self.device)
            x0[:, 2] = 0.0 #set velocity to 0.
            i0 = x0.mm(self.get_i())
            #print(x0)
            for time in range(2):
                f, r, u = self.forward(i0 + r)
        '''
        batch_size, step_num = x.size(0), x.size(1)
        act_list = []
        output_list = []
        s, h = get_items_from_dict(self.cal_init_state(x_init=x_init), ['s_init', 'h_init'])
        for time in range(x_size[1]):
            state = self.forward_once(s=s, h=h, i=i[:, time, :])
            s, u, h, o = get_items_from_dict(state, ['s', 'u', 'h', 'o'])
            act_list.append(u) # [batch_size, N_num]
            output_list.append(o) # [batch_size, output_num]
        
        #output_list = list(map(lambda x:torch.unsqueeze(x, 1), output_list))
        #act_list = list(map(lambda x:torch.unsqueeze(x, 1), act_list))

        output = torch.stack(output_list, dim=1) # [batch_size, step_num, N_num]
        act = torch.stack(act_list, dim=1) # [batch_size, step_num, N_num]
        return {
            'output': output,
            'act': act
        }
    def forward_once(self, s=None, h=None, i=None):
        batch_size = i.size(0)
        '''
        if s is None and h is None:
            s, h = self.get_init_state_zero(batch_size=batch_size)
        elif s is not None or h is not None:
            raise Exception('s and h must simultaneously be None or not None')
        '''
        noise = self.get_noise(batch_size=batch_size)
        s = (1.0 - self.time_const) * (s + noise) + self.time_const * (h + i) #x:[batch_size, sequence_length, output_num]
        s = s + noise
        u = self.act_func(s)
        '''
        if self.drop_out:
            u = self.drop_out(u)
        '''
        o = torch.mm(u, self.get_o()) # [batch_size, neuron_num] x [neuron_num, output_num]
        h = torch.mm(u, self.get_r()) + self.r_b
        return {
            's': s,
            'u': u,
            'h': h,
            'o': o,
        }
    '''
    def cal_perform_coord(self, data):
        output_truth = get_items_from_dict(data, ['output'])
        # x: [batch_size, step_num, input_num]
        # y: [batch_size, step_num, output_num]
        output, act = get_items_from_dict(self.forward(data), ['output', 'act'])
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
        loss_weight = self.weight_coeff * ( torch.mean(self.get_o() ** 2) + torch.mean(self.get_i() ** 2) )
        self.perform_list['weight'] += loss_weight.item()
        self.perform_list['act'] += loss_act.item()
        self.perform_list['coord'] += + loss_coords.item()
        self.batch_count += 1
        #self.sample_count += self
        return loss_coords + loss_act + loss_weight
    def cal_perform_pc(self, data):
        x, x_init, output_truth = get_items_from_dict(data, ['input', 'input_init', 'output'])
        batch_size = x.size(0)
        output, act = get_items_from_dict(self.forward(data), ['output', 'act'])
        
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
        #loss_weight_0 = self.weight_coeff * torch.mean(self.get_r() ** 2)
        #print(loss_weight_0)
        # dynamically alternate weight coefficient
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
        y = get_items_from_dict(data, ['output'])
        batch_size = y.size(0)
        output, act = get_items_from_dict(self.forward(data), ['output', 'act'])
        self.dict['act_avg'] = torch.mean(torch.abs(act))
        pc_output = self.place_cells.get_act(y)
        loss_coords =self.main_coeff_pc * F.mse_loss(output[:, :, 0:2], pc_output, reduction='mean')
        loss_main = self.main_coeff_coords * F.mse_loss(output[:, :, 2:-1], pc_output, reduction='mean')
        self.perform_list['pc_error_ratio'] += ( torch.sum(torch.abs(output[:, :, 2:-1] - pc_output)) / torch.sum(torch.abs(pc_output)) ).item() #relative place cells prediction error
        
        loss_act = self.act_coeff * torch.mean(act ** 2)
        
        loss_weight = self.weight_coeff * ( torch.mean(self.get_o() ** 2) + torch.mean(self.get_i() ** 2) )
        
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
    def cal_loss_weight_(self):
        #if weight_cons is None:
        #    weight_cons = self.weight_cons
        weight_cons = self.weight_cons
        if self.weight_coeff == 0.0:
            return torch.tensor([0.0], device=self.device)
        else:
            loss = 0.0
            for get_weight in weight_cons:
                weight = get_weight()
                if isinstance(weight, torch.Tensor):
                    loss += torch.mean(weight ** 2)
            loss = self.weight_coeff * loss
        return loss
    def alt_pc_act_strength(self, path, verbose=True):
        pc_mean, pc_pred_mean = self.get_output_ratio_pc(path, verbose)
        act_center_0 = self.place_cells.act_center
        self.place_cells.act_center = act_center_1 = 1.0 * act_center_0 * pc_pred_mean / pc_mean
        if verbose:
            print('alternating pc peak activation from %.3e to %.3e'%(act_center_0, act_center_1))
    def get_output_ratio_pc(self, path, verbose):
        x, y = self.prep_path(path)
        pc_output = self.place_cells.get_act(y)
        output, act = get_items_from_dict(self.forward(x), ['output', 'act'])
        pc_mean = torch.mean(pc_output).item()
        pc_pred_mean = torch.mean(output).item()
        if verbose:
            print('pc_act mean: %.3e pc_act_pred_mean: %.3e'%(pc_mean, pc_pred_mean))
        return pc_mean, pc_pred_mean
    def get_output_from_act(self, act, to_array=True):
        if isinstance(act, np.ndarray):
            # isinstance(act, type(np.ndarray)) does not work.
            act = torch.from_numpy(act).to(self.device).float()
        #print(act.size())
        output = torch.mm(act, self.get_o())
        if to_array:
            output = output.detach().cpu().numpy()
        return output

    def save(self, save_path, save_name):
        ensure_path(save_path)
        #self.update_before_save()
        with open(save_path + save_name, 'wb') as f:
            self.to(torch.device('cpu'))
            torch.save(self.dict, f)
            self.to(self.device)
    def get_weight(self, name, detach=True):
        if name in ['r']:
            w = self.get_r()
        elif name in ['o']:
            w = self.get_o()
        else:
            raise Exception('Invalid weight name: %s'%name)
        if detach:
            w = w.detach()
        return w
    def report_weight_update(self):
        for name, value in self.named_parameters():
            #if name in ['encoder.0.weight','encoder.2.weight']:
            if True:
                #print('name: {0},\t grad: {1}'.format(name, value.requires_grad))
                #print(value)
                #print(value.grad)
                value_np = value.detach().cpu().numpy()
                if self.cache.get(name) is not None:  
                    #print('  change in %s: '%name, end='')
                    #print(value_np - self.cache[name])
                    print('  ratio change in %s: '%name, end='')
                    print( np.sum(np.abs(value_np-self.cache[name])) / np.sum(np.abs(self.cache[name])) )

                self.cache[name] = value_np
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
    def plot_recurrent_weight(self, ax, cmap):
        weight_r = self.get_r().detach().cpu().numpy()
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
            
            ax.tick_params(axis='both', which='minor', length=0)

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

        # plot recurrent weight
        ax = axes[0, 0] # raises error is row_num==col_num==1
        
        self.plot_recurrent_weight(ax, cmap)

        # plot input_weight
        if self.init_method in ['linear']:
            ax = axes[0, 1]
        elif self.init_method in ['mlp']:
            pass
        else:
            pass

        plt.tight_layout()
        if save:
            ensure_path(save_path)
            plt.savefig(save_path + save_name)
    def reset_zero(self, **kw):
        batch_size = kw['i0'].size(0)
        self.x = torch.zeros((batch_size, self.N_num), device=self.device) # [batch_size, N_num]
        return 0.0 # r0       
    def reset_linear(self, **kw):
        self.x = torch.mm(kw['i0'], self.i_0_x) #[batch_size, input_num] x [input_num, N_num] = [batch_size, N_num]
        r0 = torch.mm(kw['i0'], self.i_0_r)
        return r0 # r0
    def reset_encoder(self, **kw):
        #print(kw['i0'].dtype)
        self.x_r = self.encoder(kw['i0'])
        self.x = self.x_r[:, 0:self.N_num]
        return self.x_r[:, self.N_num:] # r0
    def reset_fixed(self, **kw):
        x0 = torch.unsqueeze(self.x0, 0) # [1, input_num]
        self.x = torch.cat([x0 for i in range(kw['i0'].size(0))], dim = 0) # [batch_size, input_num]
        r0 = torch.unsqueeze(self.r0, 0) # [1, input_num]
        return torch.cat([r0 for i in range(kw['i0'].size(0))], dim = 0) # [batch_size, input_num]
    def reset_from_given(self, **kw):
        self.x = kw['x0']
    def get_noise_gaussian(self, batch_size):
        noise = torch.zeros((batch_size, self.dict['input_num']), device=self.device)
        torch.nn.init.normal_(noise, 0.0, self.noise_coeff)
        return noise


