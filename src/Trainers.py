from typing import Any

from utils import ensure_dir, get_from_dict, set_instance_variable, search_dict


class Evaluator():
    def __init__(self, dict_={}, options=None):
        if options is not None:
            self.receive_options(options)
        self.dict = dict_

    def bind_data_loader(self, data_loader):
        self.data_loader = data_loader

    def bind_model(self, model):
        self.model = model

    def receive_options(self, options):
        self.options = options
        self.device = self.options.device

    def evaluate(self):
        # evaluate model
        train_loader, test_loader = self.data_loader.get_loader()
        self.model.reset_perform()
        for data in list(train_loader):
            inputs, labels = data
            self.model.get_perform(inputs.to(self.device), labels.to(self.device))
        test_perform = self.model.report_perform(prefix='test: ')

class Trainer:
    def __init__(self, dict_, load=False, options=None):
        if options is not None:
            self.receive_options(options)

        self.dict = dict_
        set_instance_variable(self, self.dict)

        if not hasattr(self, 'anal_path'):
            self.anal_path = self.dict['anal_path'] = '../anal/'

        '''
        self.epoch_index = get_from_dict(self.dict, 'epoch_index', default=self.epoch_start, write_default=True)
        self.epoch_start = get_from_dict(self.dict, 'epoch_start', default=1, write_default=True)
        self.epoch_end = get_from_dict(self.dict, 'epoch_end', default=self.epoch_um, write_default=True)
        '''
        self.epoch_index = 0
        self.epoch_end = self.epoch_num - 1

        # save directory setting
        self.save_path_model = search_dict(self.dict, ['save_model_path', 'save_dir_model'], default='./SavedModels/',
                                           write_default=True, write_default_key='save_model_path')
        ensure_dir(self.save_path_model)

        self.save_model = get_from_dict(self.dict, 'save_model', default=True, write_default=True)
        self.save_after_train = get_from_dict(self.dict, 'save_after_train', default=True, write_default=True)
        self.save_before_train = get_from_dict(self.dict, 'save_before_train', default=True, write_default=True)
        self.anal_before_train = get_from_dict(self.dict, 'anal_before_train', default=True, write_default=True)

        if self.save_model:
            self.save_interval = search_dict(self.dict, ['save_interval', 'save_model_interval'], default=int(self.epoch_num / 10), write_default=True)

        if options is not None:
            self.options = options
            self.set_options()

        self.test_performs = self.dict['test_performs'] = {}
        self.train_performs = self.dict['train_performs'] = {}

    def train(self, report_in_batch=None, report_interval=None):
        if report_in_batch is None:
            if not hasattr(self, 'report_in_batch'):
                report_in_batch = True
            else:
                report_in_batch = self.report_in_batch
        if report_in_batch:
            if report_interval is None:
                if not hasattr(self, 'report_interval'):    
                    report_interval = int(self.batch_num / 40)
                else:
                    report_interval = self.report_interval

        if self.save_before_train:
            self.model.save(self.save_path_model, self.model.dict['name'] + '_epoch=beforeTrain')

        if self.anal_before_train:
            self.anal()

        self.optimizer.update_before_train()

        while self.epoch_index <= self.epoch_end:
            print('epoch=%d' % self.epoch_index, end=' ')
            # train model
            self.model.reset_perform()
            #batch_num = 0
            for batch_index in range(self.batch_num):
                # prepare_data
                '''
                path = self.agent.walk_random(num=self.batch_size)
                self.optimizer.train(path)
                '''
                self.agent.train(self.batch_size)
                if report_in_batch:
                    if batch_index % report_interval == 0:
                        print('batch=%d/%d' % (batch_index, self.batch_num))
                        self.model.report_perform()
                        self.model.reset_perform()
                        print('lr: %.3e'%self.optimizer.get_lr())
                #batch_num += 1
            train_perform = self.model.report_perform(prefix='train: ')
            
            '''
            # evaluate model
            self.model.reset_perform()
            for data in list(train_loader):
                inputs, labels = data
                self.optimizer.evaluate(inputs.to(self.device), labels.to(self.device))
            test_perform = self.model.report_perform(prefix='test: ')            
            self.test_performs[self.epoch_index] = test_perform
            '''

            #print('save_model:%s save_interval:%d'%(self.save_model, self.save_interval))
            if self.save_model and self.epoch_index % self.save_interval == 0:
                print('saving_model at this epoch')
                self.model.save(self.save_path_model, self.model.dict['name'] + '_epoch=%d' % self.epoch_index)
            
            if self.anal_model and self.epoch_index % self.anal_interval == 1:
                self.anal()
            
            self.optimizer.update_epoch()
            self.epoch_index += 1
        if self.save_after_train:
            self.model.save(self.save_path_model, self.model.dict['name'] + '_epoch=afterTrain')

    def receive_options(self, options):
        self.options = options
        self.device = options.device
        self.optimizer = options.optimizer
        self.agent = options.agent
        #self.model = options.model
        self.arenas = options.arenas
    def bind_agent(self, agent):
        self.agent = agent
    def bind_model(self, model):
        self.model = model
    def bind_optimizer(self, optimizer):
        self.optimizer = optimizer
    def anal(self):
        save_path = self.anal_path + 'epoch=%d/'%(self.epoch_index)
        print('Analying...epoch=%d'%(self.epoch_index))
        print('plotting path.')
        self.agent.plot_path(save_path=save_path, save_name='path_plot.png', model=self.model, plot_num=2)
        print('plotting act map.')
        self.agent.anal_act(save_path=save_path,
                            model=self.optimizer.model,
                            trainer=self,
                            arena=self.agent.arenas.current_arena(),
                            separate_ei=self.optimizer.model.separate_ei
                        )