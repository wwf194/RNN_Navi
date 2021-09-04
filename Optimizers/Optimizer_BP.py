
import torch

from utils import search_dict, GetItemsFromDict
import model
from Optimizers.Optimizer import *

class Optimizer_BP(Optimizer):
    def __init__(self, dict_=None, load=False, model=None, params=None):
        super().__init__(dict_, load)
        self.Getlr = self.Getcurrent_lr
        self.load = load
        #self.optimizer = self.build_optimizer(load=load, model=model, params=params)
        if params is not None:
            self.params = params
        elif model is not None:
            self.model = model
        self.build_optimizer(model=model, params=params, load=load)
        self.build_scheduler(load=load) # scheduler should be built after optimizer
    def bind_model(self, model, load=False):
        self.build_optimizer(model=model, load=load)
        self.build_scheduler(load=load)
    def build_optimizer(self, load=False, model=None, params=None):
        self.optimizer = model.build_optimizer(self.dict['optimizer'], params=params, model=model, load=load)
    def build_scheduler(self, load=False, verbose=False):
        #self.lr_decay = self.dict['lr_decay']
        print(self.dict.keys())
        self.scheduler = model.build_scheduler(self.dict['scheduler'], optimizer=self.optimizer, load=load)
        scheduler_type = search_dict(self.dict['scheduler'], ['type', 'method'], default='None', write_default=True)
        if verbose:
            print('Optimizer_BP.lr_decay_method: %s'%scheduler_type)
        if scheduler_type is None or scheduler_type in ['None', 'none']:
            self.update_lr = self.update_lr_none
        elif scheduler_type in ['exp']:
            self.update_lr = self.update_lr_
        elif scheduler_type in ['stepLR', 'exp_interval']:
            self.update_lr = self.update_lr_
        elif scheduler_type in ['Linear', 'linear']:
            self.update_lr = self.update_lr_
        else:
            raise Exception('build_scheduler: Invalid lr decay method: '+str(scheduler_type))
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
        self.update_after_epoch_init()
    def train(self, data):
        self.optimizer.zero_grad()
        loss = GetItemsFromDict(data, ['loss'])
        #loss = results['loss']
        loss.backward()
        self.optimizer.step()
        #self.optimizer.zero_grad() # grad should be maintained, in order to do analysis based on gradient
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
    def Getcurrent_lr(self):
        return self.optimizer.state_dict()['param_groups'][0]['lr']

