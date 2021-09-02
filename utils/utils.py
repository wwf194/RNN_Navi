#from utils_torch.utils import *
#from utils_torch import Getbest_gpu

import os
import time
import logging
import json5
import utils

ArgsGlobal = utils.json.JsonObj2PyObj({
    "ConfigDicts":{},
    "ParamDicts":{},
    "ObjRoot":{}
})

def Getdevice(args):
    if hasattr(args, 'device'):
        if args.device not in [None, 'None']:
            print(args.device)
            return args.device
    return Getbest_gpu()

def Getrequired_file(file_start):
    if isinstance(file_start, str):
        file_start = [file_start]
    elif isinstance(file_start, tuple) or isinstance(file_start, set):
        file_start = list(file_start)

    file_list = file_start
    for file in file_start:
        Getrequired_file_recur(file, file_list)
    return file_list

def Getrequired_file_recur(file, file_list):
    # file_list: stores required files in format of relative path to ./
    if not file.startswith('/') and not file.startswith('./'):
        file = './' + file
    if not file in file_list:
        file_list.append(file)
    File = import_file(file)

    if 'file_required' in dir(File):
        file_required = File.file_required
        #print(type(file_required))
        if isinstance(file_required, list):
            pass
        elif isinstance(file_required, dict):
            file_required = file_required.values()
        elif isinstance(file_required, set) or isinstance(file_required, tuple):
            file_required = list(file_required)
        else:
            raise Exception('Getrequired_file_recur: Unknown file_required type: %s'%type(file_required))
        
        for file_rel in file_required:
            file_rel_main = cal_path_rel_main(path_rel=file_rel, path_start=File.__file__, path_main=__file__)
            #print('file_rel_main: %s'%file_rel_main)
            Getrequired_file_recur(file_rel_main, file_list)


def SetLoggerGlobal():
    ArgsGlobal.logger_global = Getlogger('log-global')

def GetLoggerGlobal():
    return ArgsGlobal.logger_global

def Getlogger(logger_name='log'):
    # 输出到console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG) # 指定被处理的信息级别为最低级DEBUG，低于level级别的信息将被忽略
    if not os.path.exists("./log/"):
        os.mkdir("./log/")
    # 输出到file
    file_handler = logging.FileHandler("./log/%s-%s.txt"%(logger_name, GetTime("%Y-%m-%d-%H:%M:%S")), mode='w', encoding='utf-8')  # 不拆分日志文件，a指追加模式,w为覆盖模式
    file_handler.setLevel(logging.DEBUG)            
    logger = logging.getLogger("LoggerMain")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

def AddLog(log, logger=None, time_stamp=True):
    if logger is None:
        logger = GetLoggerGlobal()
    if time_stamp:
        logger.debug("[%s]%s"%(GetTime(), log))
    else:
        logger.debug("%s"%log)

def AddWarning(log, logger=None, time_stamp=True):
    if logger is None:
        logger = GetLoggerGlobal()
    if time_stamp:
        logger.warning("[%s][WARNING]%s"%(GetTime(), log))
    else:
        logger.warning("%s"%log)

def add_error(log, logger=None, time_stamp=True):
    if logger is None:
        logger = GetLoggerGlobal()
    if time_stamp:
        logger.error("[%s][ERROR]%s"%(GetTime(), log))
    else:
        logger.error("%s"%log)

def GetTime(format="%Y-%m-%d %H:%M:%S", verbose=False):
    time_str = time.strftime(format, time.localtime()) # Time display style: 2016-03-20 11:45:39
    if verbose:
        print(time_str)
    return time_str


# basic Json manipulation methods
def JsonFile2JsonObj(file_path):
    with open(file_path, "r") as f:
        json_dict = json5.load(f)
    return json_dict


