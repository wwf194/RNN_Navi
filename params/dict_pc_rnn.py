epoch_num = 120
batch_num = 1000 * 2
batch_size = 200

dict_trainer = {
    'epoch_num': 120,
    'batch_num': batch_num,
    'batch_num_test': int(batch_num / 20),
    'batch_size': batch_size,
    'report_in_epoch':False,
    'report_in_epoch_interval': int(batch_num / 10),
    'save_model': True,
    'save_model_path': '../saved_models/',
    'save_model_interval': int(epoch_num / 10),
    'save_before_train': True,
    'save_after_train': True,
    'test_model': True,
    'test_before_train': True,
    'anal_model': True,
    'anal_interval': 5,
    'anal_before_train': True, # might be overwritten from args.
}


def interact(**kw):
    args = kw.get('args', None)
    if args is not None:
        if hasattr(args, 'no_anal_before_train'):
            if args.no_anal_before_train:
                dict_trainer['anal_before_train'] = False


