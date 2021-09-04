
# run this script to do different tasks
# python main.py --task copy --path ./Instances/TP-1/  //move necessary files for training and analysis to path.
# Every component, such as model, optimizer, trainer, agent, is initialized according to a dict in a .py file.
# run this script to do different tasks
# python main.py --task copy --path ./Instances/TP-1/  //move necessary files for training and analysis to path.
# Every component, such as model, optimizer, trainer, agent, is initialized according to a dict in a .py file.

import os
import sys
import argparse
import json
from typing import List
from utils.utils import ArgsGlobal
import json5
import warnings
import logging
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("task", nargs="?", default="ProcessTasks")
Args = parser.parse_args()


def main():
    if Args.task in ["CleanLog", "CleanLog", "cleanlog"]:
        CleanLog()
    elif Args.task in ["ProcessTasks"]:
        try: # catch all unhandled exceptions
            ProcessTasks()
        except Exception:
            logger.error(traceback.format_exc())
    else:
        raise Exception()

def ScanConfigFile():
    import utils
    Config = utils.json.JsonFile2PyObj("./config.jsonc")
    sys.path.append(Config.LibraryPath.utils_torch)
    import utils_torch
    utils_torch.attrs.SetAttrs(ArgsGlobal, "Config", Config)
ScanConfigFile()

import utils
from utils import AddLog, AddWarning, ArgsGlobal
utils.SetLoggerGlobal()
utils.SetSaveDir()
logger = utils.GetLoggerGlobal()


import utils_torch
from utils_torch.attrs import *

#print(utils_torch.ListAllMethodsOfModule("utils_torch.json"))

def ProcessTasks():
    import utils_torch
    utils_torch.SetLogger(utils.GetLoggerGlobal()) # Pass logger to library utils_torch
    Tasks = utils.json.JsonFile2PyObj('./task.jsonc')
    for Task in Tasks:
        utils_torch.EnsureAttrs(Task, "Args", default={})
        if Task.Type in ["AddLibraryPath"]:
            AddLibraryPath(Task.Args)
        elif Task.Type in ["LoadJsonFile"]:
            LoadJsonFile(Task.Args)
        elif Task.Type in ["ParseParamStatic", "ParseParam"]:
            ParseParamStatic(Task.Args)
        elif Task.Type in ["BuildObject"]:
            BuildObject(Task.Args)
        elif Task.Type in ["FunctionCall"]:
            utils_torch.CallFunctions(Task.Args, ObjRoot=ArgsGlobal.object)
        else:
            utils.AddWarning("Unnknown Task.Type: %s"%Task.Type)

def LoadJsonFile(Args):
    if isinstance(Args, dict):
        _LoadJsonFile(Args)
    elif isinstance(Args, list):
        for Arg in Args:
            _LoadJsonFile(Arg)

def _LoadJsonFile(Args):
    PyObj = utils_torch.json.JsonFile2PyObj(Args.FilePath)
    if not Args.MountPath.startswith("&^"):
        raise Exception()
    SetAttrs(ArgsGlobal, Args.MountPath.replace("&^", ""), PyObj)

def BuildObject(Args):
    if isinstance(Args, list):
        for arg in Args:
            _BuildObject(arg)
    elif isinstance(Args, dict):
        _BuildObject(Args)

def _BuildObject(Args):
    import utils_torch
    if not Args.ParamPath.startswith("&^"):
        raise Exception()
    param = GetAttrs(ArgsGlobal, Args.ParamPath.replace("&^", ""))
    Module = utils_torch.ImportModule(Args.ModulePath)
    Obj = Module.__MainClass__(param)
    utils_torch.MountObj(Obj, ArgsGlobal, Args.MountPath.replace("&^", ""))

def LoadParamFile(Args):
    if isinstance(Args, dict):
        _LoadParamFile(Args)
    elif isinstance(Args, list):
        for Args_dict in Args:
            _LoadParamFile(Args_dict)
    else:
        raise Exception()

def _LoadParamFile(Args):
    import utils_torch
    ParamPyObj = utils_torch.JsonFile2PyObj(Args.path)
    setattr(utils.ArgsGlobal.ParamDicts, Args.name, ParamPyObj)
    AddLog("Loading parameter file %s to parameter %s"%(Args.path, Args.name))

def LoadConfigFile(Args):
    if isinstance(Args, list):
        for Args_dict in Args:
            _LoadConfigFile(Args_dict)
    else:
        _LoadConfigFile(Args)

def _LoadConfigFile(Args):
    path, name = Args["path"], Args["name"]
    utils.ArgsGlobal["ConfigDicts"][name] = utils.JsonFile2JsonObj(path)
    AddLog("Loaded configuration file from to %s config %s."%(path, name))

def AddLibraryPath(Args):
    if isinstance(Args, dict):
        _AddLibraryPath(Args)
    elif isinstance(Args, list):
        for Args_dict in Args:
            _AddLibraryPath(Args_dict)
    else:
        raise Exception()

def _AddLibraryPath(Args):
    # requires Args to be a dict.
    lib_name = Args['name']
    lib_path = Args['path']
    if lib_path=="!Getfrom_config":
        success = False
        for config_name, config_dict in utils.ArgsGlobal.ConfigDicts.__dict__.items():
            if config_dict.get("libs") is not None:
                libs = config_dict["libs"]
                if libs.get(lib_name) is not None:
                    lib_path = libs[lib_name]["path"]
                    success = True
                    break
        if not success:
            AddWarning('add_lib failed: cannot find path to lib %s'%lib_name)
            return
    if os.path.exists(lib_path):
        if os.path.isdir(lib_path):
            sys.path.append(lib_path)
            AddLog("Added library <%s> from path %s"%(lib_name, lib_path))
        else:
            AddWarning('add_lib failed: path %s exists but is not a directory.'%lib_path)
    else:
        AddWarning('add_lib: invalid lib_path: ', lib_path)

def ParseParamStatic(Args):
    import utils_torch
    utils_torch.json.PyObj2JsonFile(ArgsGlobal.param, ArgsGlobal.SaveDir + "LoadedParam")
    ArgsGlobal.param = utils_torch.parse.ParseParamPyObj(utils.ArgsGlobal.param)

    utils_torch.json.PyObj2JsonFile(ArgsGlobal.param.agent, "./agent_parsed.jsonc")
    utils_torch.json.PyObj2JsonFile(ArgsGlobal.param.model, "./model_parsed.jsonc")

def train(Args):
    if Args.type in ["SupervisedLearning"]:
        train_supervised_learning(Args)
    else:
        raise Exception()

def train_supervised_learning(Args):
    return

def CleanLog():
    import utils_torch
    utils_torch.files.RemoveAllFiles("./log/")

if __name__=="__main__":
    main()

