import torch
import torch.nn as nn
import torch.nn.functional as F

#from config import *
from utils_model import init_weight, build_mlp, get_ei_mask
from utils_model import *
from utils import get_name, get_name_args, set_instance_variable

class Neurons_LIF(nn.Module):
    def __init__(self, dict_, load=False, options=None): #input_num is neuron_num.
        super(Neurons_LIF, self).__init__()
        if options is not None:
            self.receive_options(options)
        self.dict = dict_
        set_instance_variable(self, self.dict, keys=['N_num', 'time_const', 'input_num', 'output_num', 'no_self'])

        if self.dict['separate_ei']:
            set_instance_variable(self, self.dict, keys=['E_num', 'I_num', 'Dale'])
        
        if load:
            self.f, self.r, self.b = self.dict['f'], self.dict['r'], self.dict['b']
            self.init_mode = get_name(self.dict['init_mode'])
        else:
            if self.dict['bias']:
                self.b = self.dict['b'] = torch.nn.Parameter(torch.zeros((self.dict['input_num']), device=self.device))
            else:
                self.b = self.dict['b'] = 0.0
            self.f = self.dict['f'] = self.f = nn.Parameter(torch.zeros((self.dict['input_num'], self.dict['output_num']), device=self.device, requires_grad=True))
            self.r = self.dict['r'] = self.r = nn.Parameter(torch.zeros((self.dict['input_num'], self.dict['input_num']), device=self.device, requires_grad=True))
            init_weight(self.r, self.dict['init_weight']['r'])
            init_weight(self.f, self.dict['init_weight']['f'])              
        
        self.init_mode, init_args = get_name_args(self.dict['init_mode'])
        print('inide_method: %s'%self.init_mode)
        if self.init_mode in ['zero']:
            self.reset = self.reset_zero
        elif self.init_mode in ['fixed']:
            '''
            if load:
                self.x0 = nn.Parameter(self.dict['x0'])
            else:
                self.x0 = self.dict['x0'] = torch.nn.Parameter(torch.zeros((self.dict['input_num']), device=self.device, requires_grad=True))
                torch.nn.init.normal_(self.x0, 0.0, 1.5 / self.dict['input_num']) # init input weight to be N(0, 1.5/N_num)
            '''
            if load:
                self.x_0 = nn.Parameter(self.dict['x_0'])
                self.r_0 = nn.Parameter(self.dict['r_0'])
            else:
                self.x_0 = self.dict['x_0'] = nn.Parameter(torch.zeros((self.dict['N_num']), device=self.device, requires_grad=True))
                self.r_0 = self.dict['r_0'] = nn.Parameter(torch.zeros((self.dict['N_num']), device=self.device, requires_grad=True))
                init_weight(self.r_0, self.dict['init_weight']['x_0'])
                init_weight(self.r_0, self.dict['init_weight']['r_0'])    
            self.reset = self.reset_fixed
        elif self.init_mode in ['linear']:
            if load:
                self.i_0_x = nn.Parameter(self.dict['i_0_x'])
                self.i_0_r = nn.Parameter(self.dict['i_0_r'])
            else:
                self.i_0_x = self.dict['i_0_x'] = nn.Parameter(torch.zeros((self.dict['i0_size'], self.dict['N_num']), device=self.device, requires_grad=True))
                self.i_0_r = self.dict['i_0_r'] = nn.Parameter(torch.zeros((self.dict['i0_size'], self.dict['N_num']), device=self.device, requires_grad=True))
                init_weight(self.i_0_x, self.dict['init_weight']['i_0'])
                init_weight(self.i_0_r, self.dict['init_weight']['i_0'])       
            
            '''
            if'i_0_x' in positive_weight:
                self.get_i_0_x = lambda:torch.abs(self.i_0_x)
            else:
                self.get_i_0_x = lambda:self.i_0_x
            if 'i_0_r' in positive_weight:
                self.get_i_0_r = lambda:torch.abs(self.i_0_r)
            else:
                self.get_i_0_r = lambda:self.i_0_r
            '''
            
            self.reset = self.reset_linear
        elif self.init_mode in ['given']:
            self.reset = self.reset_from_given
        elif self.init_mode in ['mlp']:
            self.encoder = build_mlp(init_args, load=load)
            self.reset = self.reset_encoder
            self.add_module('encoder', self.encoder)
        else:
            raise Exception('Neurons_LIF: Invalid init_mode:'+str(self.init_mode))

        #set recurrent
        if self.no_self:
            self.r_self_mask = torch.ones( (self.N_num, self.N_num), device=self.device, requires_grad=False )
            for i in range(self.dict['N_num']):
                self.r_self_mask[i][i] = 0.0
            self.get_r_noself = lambda :self.r * self.r_self_mask
        else:
            self.get_r_noself = lambda :self.r
        self.ei_mask = None
        if self.dict['separate_ei'] and 'r' in self.Dale:
            self.ei_mask = get_ei_mask(E_num=self.dict['E_num'], N_num=self.dict['N_num'], device=self.device)
            self.get_r_ei = lambda :torch.mm(self.ei_mask, torch.abs(self.get_r_noself()))
        else:
            self.get_r_ei = self.get_r_noself
        
        disable_connection = self.dict.setdefault('disable_connection', [])
        print('disable_connection:'+str(disable_connection))
    
        #input()
        if len(disable_connection)>0:
            self.mask = torch.ones( (self.N_num, self.N_num), device=self.device, requires_grad=False )
            if 'E->E' in disable_connection:
                print('bannning E->E connection')
                self.mask[0:self.E_num, 0:self.E_num] = 0.0
            if 'E->I' in disable_connection:
                print('bannning E->I connection')
                self.mask[0:self.E_num, self.E_num:self.N_num] = 0.0
            if 'I->E' in disable_connection:
                print('bannning I->E connection')
                self.mask[self.E_num:self.N_num, 0:self.E_num] = 0.0
            if 'I->I' in disable_connection:
                print('bannning I->I connection')
                self.mask[self.E_num:self.N_num, self.E_num:self.N_num] = 0.0
            self.get_r_mask = lambda :self.mask * self.get_r_ei()
        else:
            self.get_r_mask = self.get_r_ei
        
        self.get_r = self.get_r_mask

        #print(self.get_r())
        #input()
        #set forward weight
        if(self.dict['separate_ei'] and 'f' in self.dict['Dale']): #set mask for EI separation
            if(self.ei_mask is None):
                self.ei_mask = get_ei_mask(E_num=self.dict['E_num'], N_num=self.dict['N_num'], device=device)
            self.get_f_ei = lambda :torch.mm(self.ei_mask, torch.abs(self.f))
        else:
            self.get_f_ei = lambda :self.f
        self.get_f = self.get_f_ei
 
        self.noise_coeff = self.dict.setdefault('noise_coeff', 0.0)
        if self.noise_coeff==0.0:
            self.get_noise = lambda batch_size:0.0
        else:
            self.get_noise = self.get_noise_gaussian
        self.act_func = get_act_func(self.dict['act_func'])
        
        self.drop_out = self.dict.setdefault('drop_out', False)
        if self.drop_out:
            self.drop_out = torch.nn.Dropout(p=0.5)
    '''
    def reset_N(self, x):
        self.N.reset(x)
    def reset_x_fixed(self, x, x_0):
        self.N.reset_x_fixed(batch_size=x.size(0))
    def reset_x_zero(self, x, x_0):
        self.N.reset_x_zero(batch_size=x.size(0))
    def reset_x_linear_pc(self, x, x_0):
        self.N.reset_x_input( self.encoder_act_func( torch.mm( torch.squeeze( self.place_cells.get_activation(torch.unsqueeze(x_0, 1)) ).float(), self.i_0_x ) ) )
    def reset_x_linear_coords(self, x, x_0):
        self.N.reset_x_input(torch.mm(x_0.float(), self.i_0_x)) 
    def reset_x_mlp_coords(self, x, x_0):
        self.cache['encoder_output'] = self.encoder(x_0.float())
        #print(self.cache['encoder_output'].size())
        self.N.reset_x_input(self.cache['encoder_output'][:, 0:self.N_num])

    def get_init_r_mlp(self, x_0):
        return self.cache['encoder_output'][:, self.N_num:]
    def get_init_r_zero(self, x_0):
        return 0.0
    def get_init_r_fixed(self, x_0):
        r_0 = torch.unsqueeze(self.r_0, 0)
        #print(torch.cat([r_0 for _ in range(x_0.size(0))], dim=0).size())
        return torch.cat([r_0 for _ in range(x_0.size(0))], dim=0)
    def get_init_r_pc(self, x_0):
        return self.encoder_act_func( torch.mm( torch.squeeze( self.place_cells.get_activation(torch.unsqueeze(x_0, 1)) ).float(), self.i_0_r ) )
    def get_init_r_coords(self, x_0):
        return torch.mm(x_0.float(), self.i_0_r)
    '''
    def receive_options(self, options):
        self.options = options
        self.device = self.options.device
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
    def forward(self, i_):
        noise = self.get_noise(batch_size=i_.size(0))
        dx = i_ + self.b
        self.x = (1.0 - self.time_const) * (self.x + noise) + self.time_const * dx #x:[batch_size, sequence_length, output_num]
        u = self.act_func(self.x)
        if self.drop_out:
            u = self.drop_out(u)
        f = torch.mm(u, self.get_f()) # [batch_size, neuron_num] x [neuron_num, output_num]
        r = torch.mm(u, self.get_r())
        return f, r, u
    
    