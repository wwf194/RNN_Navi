#from utils_torch.utils import *
#from utils_torch import Getbest_gpu

import os
import time
import logging
import json5
import utils
from inspect import getframeinfo, stack

import utils_torch

GlobalParam = utils_torch.PyObj({
    "log":{}
})

def Init(SaveDirName="UnknownTask"):
    utils_torch.SetGlobalParam(GlobalParam=GlobalParam)
    utils_torch.SetSaveDir(GlobalParam=GlobalParam, Name=SaveDirName)
    utils_torch.SetLoggerGlobal(GlobalParam=GlobalParam)

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

# def GetSaveDir():
#     return GlobalParam.SaveDir

# basic Json manipulation methods
def JsonFile2JsonObj(file_path):
    with open(file_path, "r") as f:
        json_dict = json5.load(f)
    return json_dict


