
import torch

from utils import search_dict, get_items_from_dict
import utils_model
from Optimizers.Optimizer import *

class Optimizer_BP(Optimizer):
    def __init__(self, dict_=None, load=False, model=None, params=None):
        super().__init__(dict_, load)
        self.get_lr = self.get_current_lr
        self.load = load
        if params is not None:
            self.build_optimizer(load=load, model=model, params=params)
    def bind_model(self, model, load=False):
        self.build_optimizer(load=load, model=model)
    def build_optimizer(self, load=False, model=None, params=None):
        if params is not None:
            pass
        elif model is not None:
            if hasattr(model, 'get_param_to_train'):
                params = model.get_param_to_train()
        self.optimizer = utils_model.build_optimizer(self.dict['optimizer'], params=params, load=load)         
        #self.build_scheduler() # scheduler should be built after optimizer
        print(self.dict.keys())
        self.scheduler = utils_model.build_scheduler(self.dict['scheduler'], optimizer=self.optimizer, load=load)
    def build_scheduler(self, verbose=True):
        #self.lr_decay = self.dict['lr_decay']
        lr_decay = self.lr_decay = self.dict['lr_decay']
        lr_decay_method = lr_decay.get('method')
        if verbose:
            print('Optimizer_BP.lr_decay_method: %s'%lr_decay_method)
        if lr_decay_method is None or lr_decay_method in ['None', 'none']:
            self.scheduler = None
            self.update_lr = self.update_lr_none
        elif lr_decay_method in ['exp']:
            decay = search_dict(lr_decay, ['decay', 'coeff'], default=0.98, write_default=True, write_default_dict='decay')
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=decay)
            self.update_lr = self.update_lr_
        elif lr_decay_method in ['stepLR', 'exp_interval']:
            decay = search_dict(lr_decay, ['decay', 'coeff'], default=0.98, write_default=True, write_default_key='decay')
            step_size = search_dict(lr_decay, ['interval', 'step_size'], default=0.98, write_default=True, write_default_key='decay')
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, step_size=step_size, gamma=decay)
            self.update_lr = self.update_lr_
        elif lr_decay_method in ['Linear', 'linear']:
            milestones = search_dict(lr_decay, ['milestones'], throw_none_error=True)
            self.scheduler = LinearLR(self.optimizer, milestones=milestones, epoch_num=self.dict['epoch_num'])
            self.update_lr = self.update_lr_
        else:
            raise Exception('build_scheduler: Invalid lr decay method: '+str(lr_decay_method))
    def update_before_train(self):
        #print(self.dict['update_before_train'])
        self.update_before_train_items = search_dict(self.dict, ['update_before_train'], default=[], write_default=True)
        '''
        for item in self.update_before_train_items:
            if item in ['alt_pc_act_strength', 'alt_pc_strength']:
                path = self.trainer.agent.walk_random(num=self.trainer.batch_size)
                self.agent.alt_pc_act_strength(path)
            else:
                raise Exception('Invalid update_before_train item: %s'%str(item))
        '''
    def train(self, data):
        self.optimizer.zero_grad()
        loss = get_items_from_dict(self.agent.cal_perform(data), ['loss'])
        #loss = results['loss']
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
    def update_after_epoch_init(self): # decide what need to be done after every epoch 
        self.update_func_list = []
        self.update_func_list.append(self.update_lr())
    def update_after_epoch(self, **kw): # things to do after every epoch
        # in **kw: epoch_ratio: epoch_current / epoch_num_in_total
        for func in self.update_func_list:
            func(**kw)
    def update_lr_(self, **kw):
        if hasattr(self, 'scheduler'):
            self.scheduler.step()
    def update_lr_none(self, **kw):
        return
    def get_current_lr(self):
        return self.optimizer.state_dict()['param_groups'][0]['lr']

