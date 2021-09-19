
# run this script to do different tasks
# python main.py --task copy --path ./Instances/TP-1/  //move necessary files for training and analysis to path.
# Every component, such as model, optimizer, trainer, agent, is initialized according to a dict in a .py file.
# run this script to do different tasks
# python main.py --task copy --path ./Instances/TP-1/  //move necessary files for training and analysis to path.
# Every component, such as model, optimizer, trainer, agent, is initialized according to a dict in a .py file.

import os
import sys
import argparse
from typing import List

from numpy import e
from utils.utils import ArgsGlobal
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("task", nargs="?", default="ProcessTasks")
Args = parser.parse_args()

def main():
    if Args.task in ["CleanLog", "CleanLog", "cleanlog"]:
        CleanLog()
    elif Args.task in ["CleanFigure"]:
        CleanFigures()
    elif Args.task in ["ProcessTasks"]:
        ProcessTasks()
        # try: # catch all unhandled exceptions
        #     ProcessTasks()
        # except Exception:
        #     logger.error(traceback.format_exc())
        #     raise Exception()
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
utils_torch.SetLogger(utils.GetLoggerGlobal()) # Pass logger to library utils_torch
#print(utils_torch.ListAllMethodsOfModule("utils_torch.json"))

def ProcessTasks():    
    TaskPyObj = utils_torch.json.JsonFile2PyObj('./task.jsonc')
    if isinstance(TaskPyObj, list):
        TaskList = TaskPyObj
    elif isinstance(TaskPyObj, utils_torch.json.PyObj):
        TaskPyObj = utils_torch.parse.ParsePyObjStatic(TaskPyObj, ObjCurrent=TaskPyObj, ObjRoot=utils.ArgsGlobal)
        TaskPyObj = utils_torch.parse.ParsePyObjDynamic(TaskPyObj, ObjCurrent=TaskPyObj, ObjRoot=utils.ArgsGlobal)
        TaskList = TaskPyObj.Tasks
    else:
        raise Exception()
    for Index, Task in enumerate(TaskList):
        utils_torch.EnsureAttrs(Task, "Args", default={})
        if Task.Type in ["AddLibraryPath"]:
            AddLibraryPath(Task.Args)
        elif Task.Type in ["LoadJsonFile"]:
            LoadJsonFile(Task.Args)
        elif Task.Type in ["ParseParam"]:
            ParseParam(Task.Args)
        elif Task.Type in ["BuildObject"]:
            BuildObject(Task.Args)
        elif Task.Type in ["FunctionCall"]:
            utils_torch.CallFunctions(Task.Args, ObjRoot=ArgsGlobal)
        elif Task.Type in ["Train"]:
            utils_torch.train.Train(Task.Args, ObjRoot=ArgsGlobal)
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

def ParseParam(Args):
    ParseParamStatic(Args)
    utils_torch.json.PyObj2JsonFile(ArgsGlobal.param, utils_torch.RenameFileIfPathExists("param_parsed_static.jsonc"))
    ParseParamDynamic(Args)
    utils_torch.json.PyObj2JsonFile(ArgsGlobal.param, utils_torch.RenameFileIfPathExists("param_parsed_dynamic.jsonc"))
    return
def ParseParamStatic(Args):
    import utils_torch
    for attr, param in utils_torch.ListAttrsAndValues(ArgsGlobal.param):
        utils_torch.parse.ParsePyObjStatic(param, ObjCurrent=param, ObjRoot=utils.ArgsGlobal, InPlace=True)
    return
def ParseParamDynamic(Args):
    import utils_torch
    for attr, param in utils_torch.ListAttrsAndValues(ArgsGlobal.param):
        utils_torch.parse.ParsePyObjDynamic(param, ObjCurrent=param, ObjRoot=utils.ArgsGlobal, InPlace=True)
    return
def train(Args):
    if Args.type in ["SupervisedLearning"]:
        train_supervised_learning(Args)
    else:
        raise Exception()

def train_supervised_learning(Args):
    return

def CleanLog():
    import utils_torch
    utils_torch.files.RemoveAllFilesAndDirs("./log/")

def CleanFigures():
    import utils_torch
    utils_torch.files.RemoveMatchedFiles("./", r".*\.png")

if __name__=="__main__":
    main()

