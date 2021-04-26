lr = 1.0e-2

epoch_num = 120
batch_num = 1000 * 2
batch_size = 200

dict_ = {
    'train': {
        'optimizer': {
            'type': 'bp',
            'optimizer':{
                'type': 'sgd',
                'lr': lr,
            },
            'scheduler': { # used to set scheduler
                'type': 'linear',
                'milestones': [[0.50, 1.0], [0.70, 1.0e-1], [0.85, 1.0e-2], [0.95, 1.0e-3]],
            },
            'epoch_num': epoch_num
        },
        'epoch_num': epoch_num,
        'batch_num': batch_num,
        'batch_num_test': int(batch_num / 20),
        'batch_size': batch_size,
        'report_in_epoch':False,
        'report_in_epoch_interval': int(batch_num / 10),
        'save': True,
        'save_path': './saved_models/',
        'save_interval': int(epoch_num / 10),
        'save_before_train': True,
        'save_after_train': True,
        'test_model': True,
        'test_before_train': True,
        'anal': True,
        'anal_path': './anal/',
        'anal_interval': int(epoch_num / 20),
        'anal_before_train': True, # might be overwritten from args.
    }
}

def interact(env_info):
    args = env_info.get('args', None)
    if args is not None:
        if hasattr(args, 'no_anal_before_train'):
            if args.no_anal_before_train:
                if dict_.get('train') is not None:
                    dict_['train']['anal_before_train'] = False
                    dict_['train']['optimizer']['epoch_num'] = dict_['train']['epoch_num'] # some optimizers require epoch_num to set scheduler