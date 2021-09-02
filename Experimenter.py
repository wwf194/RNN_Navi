from utils import EnsurePath, GetFromDict, set_instance_variable, search_dict

class Experimenter:
    def __init__(self, dict_):
        self.dict = dict_
    def bind_agent(self, agent):
        agent = agent
    def train(self, dict_, agent=None, report_in_batch=None, report_interval=None):
        epoch_start, epoch_num, epoch_end = self.Getepoch_info(dict_)
        if agent is None:
            agent = agent

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
            agent.save(self.save_path, agent.dict['name'] + '_epoch=beforeTrain')

        if self.anal_before_train:
            self.anal(title='beforeTrain')

        self.optimizer.update_before_train()

        #print('epoch_index:%d epoch_end:%d'%(epoch_index, epoch_end))
        while epoch_index <= epoch_end:
            print('epoch=%d/%d'%(epoch_index, epoch_end), end=' ')
            # train model
            agent.reset_perform()
            #batch_num = 0
            for batch_index in range(self.batch_num):
                #print(batch_index)
                # prepare_data
                '''
                path = agent.walk_random(num=self.batch_size)
                self.optimizer.train(path)
                '''
                agent.train(self.batch_size)
                if report_in_batch:
                    if batch_index % report_interval == 0:
                        print('batch=%d/%d' % (batch_index, self.batch_num))
                        agent.report_perform()
                        agent.reset_perform()
                        print('lr: %.3e'%self.optimizer.Getlr())
                #batch_num += 1
            train_perform = agent.report_perform(prefix='train: ')

            #print('save:%s save_interval:%d'%(self.save, self.save_interval))
            if self.save_model and epoch_index % self.save_interval == 0:
                print('saving_model at this epoch')
                agent.save(self.save_path, agent.dict['name'] + '_epoch=%d' % epoch_index)
            
            if self.anal_model and epoch_index % self.anal_interval == 1:
                self.anal()
            
            self.optimizer.update_epoch()
            epoch_index += 1
        if self.save_after_train:
            agent.save(self.save_path, agent.dict['name'] + '_epoch=afterTrain')

    def __init__(self, dict_, load=False, options=None):
        if options is not None:
            self.receive_options(options)

        self.dict = dict_
        #set_instance_variable(self, self.dict)
        self.epoch_num = self.dict['epoch_num']
        self.batch_num = self.dict['batch_num']
        self.batch_size = self.dict['batch_size']

        if not hasattr(self, 'anal_path'):
            self.anal_path = self.dict.setdefault('anal_path', './anal/')

        '''
        epoch_index = GetFromDict(self.dict, 'epoch_index', default=self.epoch_start, write_default=True)
        self.epoch_start = GetFromDict(self.dict, 'epoch_start', default=1, write_default=True)
        epoch_end = GetFromDict(self.dict, 'epoch_end', default=self.epoch_um, write_default=True)
        '''
        epoch_index = 0
        epoch_end = self.epoch_num - 1

        # save directory setting
        self.save_path = search_dict(self.dict, ['save_path', 'save_model_path', 'save_dir_model'], default='./saved_models/',
                                           write_default=True, write_default_key='save_path')
        EnsurePath(self.save_path)

        self.save = search_dict(self.dict, ['save', 'save_model'], default=True, write_default=True)
        self.save_after_train = GetFromDict(self.dict, 'save_after_train', default=True, write_default=True)
        self.save_before_train = GetFromDict(self.dict, 'save_before_train', default=True, write_default=True)
        self.anal_before_train = GetFromDict(self.dict, 'anal_before_train', default=True, write_default=True)

        if self.save:
            self.save_interval = search_dict(self.dict, ['save_interval', 'save_model_interval'], default=int(self.epoch_num / 10), write_default=True)

        if options is not None:
            self.options = options
            self.set_options()

        self.test_performs = self.dict['test_performs'] = {}
        self.train_performs = self.dict['train_performs'] = {}

        self.anal_model = self.dict.setdefault('anal_model', True)

    def bind_agent(self, agent):
        agent = agent
    def bind_model(self, model):
        self.model = model
    def bind_optimizer(self, optimizer):
        self.optimizer = optimizer
    def anal(self, title=None, save_path=None, verbose=True):
        if save_path is None:
            if title is None:
                save_path = self.anal_path + 'epoch=%d/'%(epoch_index)
            else:
                save_path = self.anal_path + 'epoch=%s/'%(title)
        EnsurePath(save_path)
        agent.anal(save_path=save_path, trainer=self)
    
    def Getepoch_info(self, dict_):
        epoch_start = dict_.get('epoch_start')
        epoch_num = dict_.get('epoch_num')
        epoch_end = dict_.get('epoch_end')

        if epoch_start is not None and epoch_num is not None and epoch_end is not None:
            #epoch_start, epoch_num, epoch_end = dict_['epoch_start'], dict_['epoch_end'], dict_['epoch_end']
            if epoch_end == epoch_start + epoch_num - 1:
                pass
            else:
                raise Exception('epoch_end(%d) != epoch_start(%d) + epoch_num(%d)'%(epoch_end, epoch_start, epoch_num))
        elif epoch_start is not None and epoch_end is not None:
            epoch_num = epoch_end - epoch_start + 1
        elif epoch_start is not None and epoch_num is not None:
            epoch_end = epoch_start + epoch_num - 1
        elif epoch_end is not None and epoch_start is not None:
            epoch_start = epoch_end - epoch_num + 1
        else:
            raise Exception('At least 2 out of 3 params(epoch_start(%d), epoch_num(%d), epoch_end(%d)) must be given.'\
                %(epoch_start, epoch_num, epoch_end))
        return epoch_start, epoch_num, epoch_end