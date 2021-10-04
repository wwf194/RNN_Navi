#from utils_torch.utils import *
#from utils_torch import Getbest_gpu

import os
import time
import logging
import json5
import utils
from inspect import getframeinfo, stack

import utils_torch

ArgsGlobal = utils_torch.PyObj({
    "logger":{}
})

def Init():
    utils_torch.SetSaveDir(ArgsGlobal)
    utils_torch.SetArgsGlobal(ArgsGlobal)
    utils_torch.SetLoggerGlobal(ArgsGlobal)

def GetRequiredFile(file_start):
    if isinstance(file_start, str):
        file_start = [file_start]
    elif isinstance(file_start, tuple) or isinstance(file_start, set):
        file_start = list(file_start)

    file_list = file_start
    for file in file_start:
        _GetRequiredFile(file, file_list)
    return file_list

def _GetRequiredFile(file, file_list):
    # file_list: stores required files in format of relative path to ./
    if not file.startswith('/') and not file.startswith('./'):
        file = './' + file
    if not file in file_list:
        file_list.append(file)
    File = utils_torch.ImportFile(file)

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
            _GetRequiredFile(file_rel_main, file_list)

# class Logger:
#     def __init__(self, Name):
#         self.logger = _CreateLogger(Name)
#     def AddLog(self, log, TimeStamp=True, File=True, LineNum=True):
#         Caller = getframeinfo(stack()[1][0])
#         if TimeStamp:
#             log = "[%s]%s"%(GetTime(), log)
#         if File:
#             log = "%s File \"%s\""%(log, Caller.filename)
#         if LineNum:
#             log = "%s, line %d"%(log, Caller.lineno)
#         self.logger.debug(log)

#     def AddWarning(self, log, TimeStamp=True, File=True, LineNum=True):
#         Caller = getframeinfo(stack()[1][0])
#         if TimeStamp:
#             log = "[%s][WARNING]%s"%(GetTime(), log)
#         if File:
#             log = "%s File \"%s\""%(log, Caller.filename)
#         if LineNum:
#             log = "%s, line %d"%(log, Caller.lineno)
#         self.logger.debug(log)

#     def AddError(self, log, TimeStamp=True):
#         if TimeStamp:
#             self.logger.error("[%s][ERROR]%s"%(GetTime(), log))
#         else:
#             self.logger.error("%s"%log)

# def AddLog(log, logger=None, *args, **kw):
#     ParseLogger(logger).AddLog(log, *args, **kw)

# def AddWarning(log, logger=None, *args, **kw):
#     ParseLogger(logger).AddWarning(log, *args, **kw)

# def AddError(log, logger=None, *args, **kw):
#     ParseLogger(logger).AddError(log, *args, **kw)

# def ParseLogger(logger):
#     if logger is None:
#         logger = utils_torch.ArgsGlobal.logger.Global
#     elif isinstance(logger, str):
#         logger = GetLogger(logger)
#     else:
#         raise Exception()
#     return logger


def GetSaveDir():
    return ArgsGlobal.SaveDir

# def GetLogger(Name, CreateIfNone=True):
#     if not hasattr(ArgsGlobal.logger, Name):
#         if CreateIfNone:
#             AddLogger(Name)
#         else:
#             raise Exception()
#     return getattr(ArgsGlobal.logger, Name)
# def CreateLogger(Name):
#     return Logger(Name)

# def _CreateLogger(Name):
#     # 输出到console
#     console_handler = logging.StreamHandler()
#     console_handler.setLevel(logging.DEBUG) # 指定被处理的信息级别为最低级DEBUG，低于level级别的信息将被忽略
#     if not os.path.exists("./log/"):
#         os.mkdir("./log/")
    
#     if not hasattr(ArgsGlobal, "SaveDir"):
#         SetSaveDir()

#     # 输出到file
#     file_handler = logging.FileHandler(ArgsGlobal.SaveDir + "%s.txt"%(Name), mode='w', encoding='utf-8')  # 不拆分日志文件，a指追加模式,w为覆盖模式
#     file_handler.setLevel(logging.DEBUG)            
#     logger = logging.Logger(Name)
#     logger.setLevel(logging.DEBUG)
#     logger.addHandler(console_handler)
#     logger.addHandler(file_handler)
#     return logger

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


