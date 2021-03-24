import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib as mpl

from utils import set_instance_variable, contain, get_name
from utils_anal import *
from utils_model import *
from utils_plot import norm_and_map
from utils_model import init_weight
from Place_Cells import Place_Cells
from Neurons_LIF import Neurons_LIF

class RNN_Navi(nn.Module):
    def __init__(self, dict_=None, load=False, f=None, options=None):
        super(RNN_Navi, self).__init__()
        self.dict = dict_
        self.options = options
        set_instance_variable(self, self.dict)
        # for compatibility with older versions
        if self.input_mode not in ['v_xy', 'v_hd']: 
            self.input_mode = 'v_xy'
        if not hasattr(self, 'init_mode'):
            self.init_mode = self.dict['init_mode'] = self.dict['init_method']
        self.load = load

        if self.load:
            if f is not None:
                self.dict = torch.load(f, map_location=self.device) 
            else:
                self.dict = dict_
        else:
            self.dict = dict_

    def receive_options(self, options):
        self.options = options
        self.device = self.options.device
        self.options = options
        
        self.dict_N = self.dict['N']
        if self.target in ['pc', 'pc_coords', 'coords_pc']:
            self.place_cells = Place_Cells(dict_ = self.dict['place_cells'], options=options, load=self.load)
            self.place_cells.receive_options(options)
            self.pc_num = self.dict['place_cells']['N_num']
            
        if self.target in ['pc']:
            self.get_perform = self.get_loss = self.get_perform_pc
            self.perform_list = {'pc':0.0, 'act':0.0, 'weight':0.0}
            self.dict_N['i0_size'] = self.pc_num
            self.prepare_x0 = self.prepare_x0_pc
            self.output_mode = self.dict['output_mode'] = 'pc'
        elif self.target in ['pc_coords', 'coords_pc']:
            self.place_cells = Place_Cells(dict_ = self.dict['place_cells'], options=options, load=self.load)
            self.get_perform = self.get_loss = self.get_perform_pc_coords
            self.perform_list = {'pc':0.0, 'coords':0.0, 'act':0.0, 'weight':0.0}
            self.dict_N['i0_size'] = self.pc_num + self.input_num
            self.output_mode = self.dict['output_mode'] = 'pc_coords'
        elif self.target in ['coords']:
            self.get_perform = self.get_loss = self.get_perform_coords
            self.perform_list = {'coords':0.0, 'act':0.0, 'weight':0.0}
            self.dict_N['i0_size'] = self.input_num
            self.prepare_x0 = self.prepare_x0_null
            self.output_mode = self.dict['output_mode'] = 'coords'
        else:
            raise Exception('RNN_Navi: Invalid target: %s'%str(target))

        self.dict_loss = self.dict['loss']
        self.main_loss = self.dict_loss['main']['type']
        self.main_coeff = self.dict_loss['main']['coeff']
        self.act_coeff = self.dict_loss['act']
        self.weight_coeff = self.dict_loss['weight']
        self.dynamic_weight_coeff = self.dict_loss['dynamic_weight_coeff']['enable']
        if self.dynamic_weight_coeff:
            self.alt_weight_coeff_count = 0
            self.weight_ratio = self.dict_loss['dynamic_weight_coeff']['ratio_to_main_loss']
            self.weight_ratio_min = self.dict_loss['dynamic_weight_coeff']['ratio_to_main_loss_min']
            self.weight_ratio_max = self.dict_loss['dynamic_weight_coeff']['ratio_to_main_loss_max']
            #self.weight_coeff_0 = self.dict_loss['weight_coeff_0'] = self.weight_coeff

        self.init_perform()
        self.N = Neurons_LIF(dict_=self.dict_N, options=options, load=self.load) #neurons
        set_instance_variable(self, self.dict, keys=['output_num'])

        if self.load:
            self.i = torch.nn.Parameter(self.dict['i']) #input weight
            self.init_mode = get_name(self.dict['init_mode'])
        else:
            self.dict_N = self.N.dict
            self.dict['i'] = self.i = torch.nn.Parameter(torch.zeros((self.dict['input_num'], self.dict['N_num']), device=self.device, requires_grad=True)) #input weights
            init_weight(self.i, self.dict['init_weight']['i'])
            self.init_mode = get_name(self.dict['init_mode'])

        # print(list(self.encoder.parameters())[0])
        # for name, value in self.named_parameters():
        #     print('name: {0},\t grad: {1}'.format(name, value.requires_grad))
        #input()

        self.time_const = self.N.time_const
        self.b = self.N.b
        self.get_r = self.N.get_r
        self.get_i = lambda:self.i

        if self.target in ['pc', 'pc_coords']:
            self.dict['pc_error'] = 0.0
        self.batch_count = 0
        self.cache = {}
        self.N_num = self.N.dict['N_num']
        if self.N.dict['separate_ei']:
            self.E_num = self.N.dict['E_num']
            self.I_num = self.N.dict['I_num']
        

        
    def update_before_save(self):
        if get_name(self.dict['init_mode']) in ['mlp']:
            update_mlp(self.dict['encoder_dict'], self.encoder_layers)
    def forward(self, x): #([batch_size, step_num, input_num], [batch_size, input_num])
        x0 = x[1].to(self.device) # [batch_size, input_num]
        x = x[0].to(self.device) # [batch_size, step_num, input_num]
        x_size = list(x.size())
        #i_ = x.mm(self.get_i()) #[batch_size, step_num, N_num]
        x = x.view(x_size[0] * x_size[1], x_size[2])
        i_ = ( x.mm(self.get_i()) ) # [batch_size, step_num, N_num]
        i_ = i_.view(x_size[0], x_size[1], -1)
        r0 = self.N.reset(x=x, i0=self.prepare_x0(x0))
        '''
        if pre_input:
            x0 = torch.squeeze(x[:, 0, :]).detach().cpu().numpy() #(batch_size, input_num)
            x0 = torch.from_numpy(x0).to(self.device)
            x0[:, 2] = 0.0 #set velocity to 0.
            i0 = x0.mm(self.get_i())
            #print(x0)
            for time in range(2):
                f, r, u = self.N.forward(i0 + r)
        '''
        act_list = []
        output_list = []
        r = r0
        for time in range(x_size[1]):
            f, r, u = self.N.forward(torch.squeeze(i_[:, time, :]) + r)
            act_list.append(u) # [batch_size, N_num]
            output_list.append(f) # [batch_size, output_num]
        
        output_list = list(map(lambda x:torch.unsqueeze(x, 1), output_list))
        act_list = list(map(lambda x:torch.unsqueeze(x, 1), act_list))
        #print(list(output_list[0].size()))
        output = torch.cat(output_list, dim=1)
        act = torch.cat(act_list, dim=1)
        return output, act
        #return output_list, act_list
    def prepare_x0_pc(self, x0): # [batch_size, input_num]
        #return torch.squeeze(self.place_cells.get_act(torch.unsqueeze(x0, 1))) # [batch_size, pc_num]
        # to be implemented: batch_size must be > 1.
        return torch.squeeze(self.place_cells.get_act(torch.unsqueeze(x0, 1))).float() # [batch_size, pc_num]
    def prepare_x0_null(self, x0):
        return x0 # [batch_size, input_num]
    def report_perform(self, prefix=''):
        print(prefix, end='')
        for key in self.perform_list.keys():
            #print('%s:%s'%(key, str(self.perform_list[key]/self.batch_count)), end=' ')
            print('%s:%.4e'%(key, self.perform_list[key]/self.batch_count), end=' ')
        print('\n')
    def init_perform(self):
        self.batch_count = 0
        self.sample_count = 0
        if self.get_perform == self.get_perform_pc:
            self.perform_list['pc_error_ratio'] = 0.0
    def reset_perform(self):
        self.batch_count = 0
        self.sample_count = 0
        #print(self.batch_count)
        for key in self.perform_list.keys():
            self.perform_list[key] = 0.0
    def get_perform(self):
        return # to be implemented
    def get_perform_coords(self, path):
        x, y = self.prepare_path(path)
        #x: [batch_size, sequence_length, input_num]
        #y: [batch_size, sequence_length, output_num]
        output, act = self.forward(x)
        self.dict['act_avg'] = torch.mean(torch.abs(act))
        loss_coords = self.main_coeff * F.mse_loss(output, y, reduction='mean')
        '''
        loss_coords = 0.0
        for time in range(x.size(1)):
            loss_coords = loss_coords + F.mse_loss(output[time], torch.squeeze(y[:, time, :]), reduction='mean')
        loss_coords = coords_index * loss_coords / (x.size(1))
        '''
        loss_act = self.act_coeff * torch.mean(act ** 2)
        '''
        loss_act = 0.0
        for time in range(x.size(1)):
            loss_act = loss_act + torch.mean(act[time] ** 2)
        loss_act = loss_act / (x.size(1))
        '''
        loss_weight = self.weight_coeff * ( torch.mean(self.N.get_f() ** 2) + torch.mean(self.get_i() ** 2) )
        self.perform_list['weight'] = self.perform_list['weight'] + loss_weight.item()
        self.perform_list['act'] = self.perform_list['act'] + loss_act.item()
        self.perform_list['coords'] = self.perform_list['coords'] + loss_coords.item()
        self.batch_count += 1
        #self.sample_count += self
        return loss_coords + loss_act + loss_weight

    def get_perform_pc(self, path):
        x, y = self.prepare_path(path)
        pc_output = self.place_cells.get_act(y)
        output, act = self.forward(x)
        
        self.dict['act_avg'] = torch.mean(torch.abs(act))
        
        self.perform_list['pc_error_ratio'] += ( torch.sum(torch.abs(output - pc_output)) / torch.sum(torch.abs(pc_output)) ).item() # relative place cells prediction error
        
        if self.main_loss in ['MSE']:
            loss_pc = self.main_coeff * F.mse_loss(output, pc_output, reduction='mean')
        elif self.main_loss in ['CEL']:
            loss_pc = - self.main_coeff * torch.mean( pc_output * F.log_softmax(output, dim=2) )
        else:
            raise Exception('Invalid main loss: %s'%str(self.main_loss))
        
        self.dict['pc_avg'] = torch.mean(torch.abs(pc_output))
        loss_act = self.act_coeff * torch.mean(act ** 2)
        
        loss_weight_0 = self.weight_coeff * torch.mean(self.get_r() ** 2)
        # dynamically alternate weight coefficient
        if self.dynamic_weight_coeff:
            loss_ratio = loss_weight_0.item() / loss_pc.item() # ratio of weight loss to pc loss.
            #print('loss_ratio:%.3f'%loss_ratio)
            if self.weight_ratio_min < loss_ratio < self.weight_ratio_max:
                loss_weight = loss_weight_0
            else:
                weight_coeff_0 = self.weight_coeff
                self.weight_coeff = self.weight_coeff * self.weight_ratio / loss_ratio #alternating weight cons index so that loss_weight == loss_pc * dynamic_weight_ratio
                self.alt_weight_coeff_count += 1
                if self.alt_weight_coeff_count > 50:
                    print('alternating weight_coeff from %.3e to %.3e'%(weight_coeff_0, self.weight_coeff))  
                    self.alt_weight_coeff_count = 0
                loss_weight = self.weight_coeff / weight_coeff_0 * loss_weight_0
        else:
            loss_weight = loss_weight_0
        
        self.perform_list['weight'] += loss_weight.item()
        self.perform_list['act'] += loss_act.item()
        self.perform_list['pc'] += loss_pc.item()
        
        self.batch_count += 1
        return loss_pc + loss_act + loss_weight
    def get_perform_pc_coords(self, path):
        x, y = self.prepare_path(path)
        output, act = self.forward(x)
        self.dict['act_avg'] = torch.mean(torch.abs(act))
        pc_output = self.place_cells.get_act(y)
        loss_coords =self.main_coeff_pc * F.mse_loss(output[:, :, 0:2], pc_output, reduction='mean')
        loss_pc = self.main_coeff_coords * F.mse_loss(output[:, :, 2:-1], pc_output, reduction='mean')
        self.perform_list['pc_error_ratio'] += ( torch.sum(torch.abs(output[:, :, 2:-1] - pc_output)) / torch.sum(torch.abs(pc_output)) ).item() #relative place cells prediction error
        
        loss_act = self.act_coeff * torch.mean(act ** 2)
        
        loss_weight = self.weight_coeff * ( torch.mean(self.N.get_f() ** 2) + torch.mean(self.get_i() ** 2) )
        
        self.perform_list['weight'] += loss_weight.item()
        self.perform_list['act'] += loss_act.item()
        self.perform_list['coords'] += loss_coords.item()
        self.perform_list['pc'] += loss_pc.item()
        
        self.batch_count += 1
        return loss_pc + loss_coords + loss_act + loss_weight
    def alt_pc_act_strength(self, path, verbose=True):
        pc_mean, pc_pred_mean = self.get_output_ratio_pc(path, verbose)
        act_center_0 = self.place_cells.act_center
        self.place_cells.act_center = act_center_1 = 1.0 * act_center_0 * pc_pred_mean / pc_mean
        if verbose:
            print('alternating pc peak activation from %.3e to %.3e'%(act_center_0, act_center_1))
    def get_output_ratio_pc(self, path, verbose):
        x, y = self.prepare_path(path)
        pc_output = self.place_cells.get_act(y)
        output, act = self.forward(x)
        pc_mean = torch.mean(pc_output).item()
        pc_pred_mean = torch.mean(output).item()
        if verbose:
            print('pc_act mean: %.3e pc_act_pred_mean: %.3e'%(pc_mean, pc_pred_mean))
        return pc_mean, pc_pred_mean
    
    def get_output_from_act(self, act, to_array=True):
        if type(act) is np.ndarray:
            # isinstance(act, type(np.ndarray)) does not work.
            act = torch.from_numpy(act).to(self.device).float()
        print(act.size())
        output = torch.mm(act, self.N.get_f())
        if to_array:
            output = output.detach().cpu().numpy()
        return output

    def save(self, save_path, save_name):
        ensure_path(save_path)
        self.update_before_save()
        with open(save_path + save_name, 'wb') as f:
            net = self.to(torch.device('cpu'))
            torch.save(net.dict, f)
            net = self.to(self.device)
    def get_weight(self, name, positive=True, detach=False):
        if name in ['r']:
            if positive:
                return torch.abs(self.N.get_r())
            else:
                return self.N.get_r()
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
    def prepare_path(self, path):
        if self.input_mode in ['v_xy']:
            inputs = torch.from_numpy(path['xy_delta']).float() # [batch_size, sequence_length, (vx, vy)]
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
        if self.init_mode in ['linear']:
            ax = axes[0, 1]
        elif self.init_mode in ['mlp']:
            pass
        else:
            pass

        plt.tight_layout()
        if save:
            ensure_path(save_path)
            plt.savefig(save_path + save_name)
    

