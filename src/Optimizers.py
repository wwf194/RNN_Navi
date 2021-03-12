import abc
import sys
sys.path.append('./Optimizers/')

import torch
from utils import get_from_dict, set_instance_variable, search_dict
import utils_model

from LRSchedulers import LinearLR

class Optimizer(abc.ABC): 
    def __init__(self, dict_=None, load=False, options=None):
        #if options is not None:
        #    self.receive_options(options)
        self.dict = dict_
        self.verbose = get_from_dict(self.dict, 'verbose', default=True, write_default=True)
        #set_instance_variable(self, self.dict)
    def bind_model(self, model):
        print('eee')
        if self.model is not None:
            if self.options.verbose:
                print('Optimizer: binding new model. warning: this optimizer has already bound a model, and it will be detached.')
        else:
            self.model = model
    def detach_model(self, model):
        if self.model is None:
            if self.options.verbose:
                print('Optimizer: detaching model. this optimizer hasn\'t bound a model.')
        else:
            self.model = None
    @abc.abstractmethod # must be implemented by child class.
    def train(self):
        return
    def save(self, dir_):
        with open(dir_, 'wb') as f:
            net = self.to(torch.device('cpu'))
            torch.save(net.dict, f)
            net = self.to(self.device)

class Optimizer_BP(Optimizer):
    def __init__(self, dict_=None, load=False, options=None):
        super().__init__(dict_, load, options)
    def receive_options(self, options):
        self.options = options
        self.device = options.device
        #self.model = options.model
        self.trainer = options.trainer
        #print('options.model:'+str(options.model))
        #self.build_optimizer()

    def bind_model(self, model):
        self.model = model
        self.build_optimizer()
        self.update_epoch_init()
        if self.scheduler is not None:
            self.dict['scheduler_dict'] = None
        else:
            self.dict['scheduler_dict'] = self.scheduler.state_dict()
        self.get_lr = self.get_current_lr
    def update_before_train(self):
        print(self.dict['update_before_train'])
        self.update_before_train_items = search_dict(self.dict, ['update_before_train'], default=[], write_default=True)
        
        for item in self.update_before_train_items:
            if item in ['alt_pc_act_strength', 'alt_pc_strength']:
                path = self.trainer.agent.walk_random(num=self.trainer.batch_size)
                self.model.alt_pc_act_strength(path)
            else:
                raise Exception('Invalid update_before_train item: %s'%str(item))
    def build_optimizer(self, load=False):
        self.optimizer = utils_model.build_optimizer(self.dict['optimizer'], model=self.model, load=load)
    def train(self, data):
        self.optimizer.zero_grad()
        loss = self.model.get_loss(data)
        #loss = results['loss']
        loss.backward()
        self.optimizer.step()
    def update_epoch_init(self): # decide what need to be done after every epoch 
        self.update_func_list = []
        self.update_func_list.append(self.update_lr_init())
    def update_epoch(self, **kw): # things to do after every epoch
        # in **kw: epoch_ratio: epoch_current / epoch_num_in_total
        for func in self.update_func_list:
            func(**kw)

    def update_lr_init(self): # define self.scheduler and return an update_lr method according to settings in self.dict_.
        #self.lr_decay = self.dict['lr_decay']
        lr_decay = self.lr_decay = self.dict['lr_decay']
        lr_decay_method = lr_decay.get('method')
        print(lr_decay_method)
        if lr_decay_method in ['None', 'none'] or lr_decay_method is None:
            
            return self.update_lr_none
        elif lr_decay_method in ['exp']:
            decay = search_dict(lr_decay, ['decay', 'coeff'], default=0.98, write_default=True, write_default_dict='decay')
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=decay)
            return self.update_lr
        elif lr_decay_method in ['stepLR', 'exp_interval']:
            decay = search_dict(lr_decay, ['decay', 'coeff'], default=0.98, write_default=True, write_default_key='decay')
            step_size = search_dict(lr_decay, ['interval', 'step_size'], default=0.98, write_default=True, write_default_key='decay')
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, step_size=step_size, gamma=decay)
            return self.update_lr
        elif lr_decay_method in ['Linear', 'linear']:
            milestones = search_dict(lr_decay, ['milestones'], throw_none_error=True)
            self.scheduler = LinearLR(self.optimizer, milestones=milestones, epoch_num=self.trainer.epoch_num)
            return self.update_lr
        else:
            raise Exception('Invalid lr decay method: '+str(lr_decay_method))
    def get_current_lr(self):
        return self.optimizer.state_dict()['param_groups'][0]['lr']
    def update_lr(self, **kw):
        self.scheduler.step()
    def update_lr_none(self, **kw):
        return
    def detach_model(self):
        self.model = None
    
    