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
    else:
        raise Exception('At least 2 out of 3 params(epoch_start(%d), epoch_num(%d), epoch_end(%d)) must be given.'\
            %(epoch_start, epoch_num, epoch_end))
    return epoch_start, epoch_num, epoch_end