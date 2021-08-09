from utils_ import search_dict, get_from_dict, ensure_path

def get_epoch_info(dict_):
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
    elif epoch_num is not None:
        epoch_start = 0
        epoch_end = epoch_num - 1
    elif epoch_end is not None:
        epoch_start = 0
        epoch_num = epoch_end + 1
    else:
        raise Exception('Invalid epoch info: epoch_start:%s, epoch_num:%s, epoch_end:%s) must be given.'\
            %(epoch_start, epoch_num, epoch_end))
    return epoch_start, epoch_num, epoch_end

def set_train_info(dict_):
    save_path = search_dict(dict_, ['save_path', 'save_model_path', 'save_dir_model'], default='./saved_models/',
                                        write_default=True, write_default_key='save_path')
    ensure_path(save_path)

    save = search_dict(dict_, ['save', 'save_model'], default=True, write_default=True)
    save_after_train = search_dict(dict_, ['save_after_train'], default=True, write_default=True)
    save_before_train = search_dict(dict_, ['save_before_train'], default=True, write_default=True)
    
    anal = search_dict(dict_, ['anal'], default=True, write_default=True)
    anal_interval = search_dict(dict_, ['anal_interval'], default=int(dict_['epoch_num'] / 20), write_default=True)
    anal_before_train = search_dict(dict_, ['anal_before_train'], default=True, write_default=True)
    anal_after_train = search_dict(dict_, ['anal_after_train'], default=True, write_default=True)

    if save:
        save_interval = search_dict(dict_, ['save_interval', 'save_model_interval'], default=int(dict_['epoch_num'] / 10), write_default=True)
    
    return {
        'save': save,
        'save_interval': save_interval,
        'save_after_train': save_after_train,
        'save_before_train': save_before_train,
        'anal': anal,
        'anal_interval': anal_interval,
        'anal_before_train': anal_before_train,
        'anal_after_train': anal_after_train,
    }