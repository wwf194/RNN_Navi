
# run this script to do different tasks
# python main.py --task copy --path ./Instances/TP-1/  //move necessary files for training and analysis to path.
# Every component, such as model, optimizer, trainer, agent, is initialized according to a dict in a .py file.
# run this script to do different tasks
# python main.py --task copy --path ./Instances/TP-1/  //move necessary files for training and analysis to path.
# Every component, such as model, optimizer, trainer, agent, is initialized according to a dict in a .py file.

import os
from re import L
import sys
import argparse
import traceback

#from utils.utils import ArgsGlobal

parser = argparse.ArgumentParser()
parser.add_argument("task", nargs="?", default="ProcessTasks")
parser.add_argument("-IsDebug", default=True)
Args = parser.parse_args()

def main():
    if Args.task in ["CleanLog", "CleanLog", "cleanlog"]:
        CleanLog()
    elif Args.task in ["CleanFigure"]:
        CleanFigures()
    elif Args.task in ["ProcessTasks"]:
        TaskObj = LoadTaskFile()
        TaskList = utils_torch.ParseTaskObj(TaskObj)
        if not Args.IsDebug:
            try: # catch all unhandled exceptions
                ProcessTasks(TaskList)
            except Exception:
                utils_torch.AddError(traceback.format_exc())
                raise Exception()
        else:
            ProcessTasks(TaskList)
    elif Args.task in ["TotalLines"]:
        utils_torch.CalculateGitProjectTotalLines()
    else:
        raise Exception("Inavlid Task: %s"%Args.task)

def ProcessTasks(TaskList):  
    for Index, Task in enumerate(TaskList):
        utils_torch.EnsureAttrs(Task, "Args", default={})
        TaskType = Task[0]
        TaskArgs = Task[1]
        # if TaskType in ["FunctionCall"]:
        #     print("aaa")
        if TaskType in ["AddLibraryPath"]:
            AddLibraryPath(TaskArgs)
        elif TaskType in ["LoadJsonFile"]:
            LoadJsonFile(TaskArgs)
        elif TaskType in ["LoadParamFile"]:
            LoadParamFile(TaskArgs)
        elif TaskType in ["ParseParam", "ParseParamStatic"]:
            ParseParamStatic(TaskArgs)
        elif TaskType in ["ParseParamDynamic"]:
            ParseParamDynamic(TaskArgs)
        elif TaskType in ["ParseSelf"]:
            utils_torch.parse.ParsePyObjStatic(TaskList, ObjRoot=utils.ArgsGlobal, InPlace=True)
        elif TaskType in ["BuildObjectFromParam"]:
            BuildObjFromParam(TaskArgs)
        elif TaskType in ["SetTensorLocation"]:
            SetTensorLocation(TaskArgs)
        elif TaskType in ["SetLogger", "SetDataLogger"]:
            SetLogger(TaskArgs)
        elif TaskType in ["FunctionCall"]:
            utils_torch.CallFunctions(TaskArgs, ObjRoot=ArgsGlobal)
        elif TaskType in ["Train"]:
            utils_torch.train.Train(TaskArgs, ObjRoot=ArgsGlobal, Logger=utils.ArgsGlobal.LoggerData)
        elif TaskType in ["DoTasks"]:
            _TaskList = utils_torch.ParseTaskObj(TaskArgs, ObjRoot=ArgsGlobal)
            ProcessTasks(_TaskList)
        else:
            utils.AddWarning("Unknown Task.Type: %s"%Task.Type)

def SetTensorLocation(Args):
    EnsureAttrs(Args, "Method", default="Auto")
    if Args.Method in ["Auto", "auto"]:
        Location = utils_torch.GetGPUWithLargestUseableMemory()
    else:
        raise Exception()

    for Obj in utils_torch.ListValues(ArgsGlobal.object):
        if hasattr(Obj, "SetTensorLocation"):
            Obj.SetTensorLocation(Location)

def SetLogger(Args):
    utils.ArgsGlobal.LoggerData = utils_torch.log.LoggerForEpochBatchTrain()

def ScanConfigFile(FilePath="./config.jsonc"):
    import json5
    with open(FilePath, "r") as f:
        JsonObj = json5.load(f) # json5 allows comments
    Config = JsonObj
    sys.path.append(Config["LibraryPath"]["utils_torch"])
    import utils_torch
    import utils
    utils_torch.attrs.SetAttrs(utils.ArgsGlobal, "Config", Config)
ScanConfigFile()

import utils
from utils import ArgsGlobal
utils.Init()

import utils_torch
from utils_torch.attrs import *
utils_torch.SetArgsGlobal(utils.ArgsGlobal)
#utils_torch.SetLogger(utils.GetLoggerGlobal()) # Pass logger to library utils_torch
#print(utils_torch.ListAllMethodsOfModule("utils_torch.json"))

def LoadTaskFile(FilePath="./task.jsonc", Save=True):
    TaskObj = utils_torch.json.JsonFile2PyObj('./task.jsonc')
    return TaskObj


def BuildObjFromParam(Args):
    if isinstance(Args, utils_torch.PyObj):
        Args = GetAttrs(Args)
    if isinstance(Args, list):
        for arg in Args:
            _BuildObjFromParam(arg)
    elif isinstance(Args, utils_torch.PyObj):
        _BuildObjFromParam(Args)
    else:
        raise Exception()

def _BuildObjFromParam(Args):
    import utils_torch
    param = utils_torch.parse.Resolve(Args.ParamPath, ObjRoot=utils.ArgsGlobal)
    Module = utils_torch.ImportModule(Args.ModulePath)
    Obj = Module.__MainClass__(param)
    utils_torch.MountObj(Obj, ArgsGlobal, Args.MountPath.replace("&^", ""))

def LoadJsonFile(Args):
    if isinstance(Args, utils_torch.PyObj):
        Args = GetAttrs(Args)
    if isinstance(Args, dict):
        _LoadJsonFile(utils_torch.json.JsonObj2PyObj(Args))
    elif isinstance(Args, list):
        for Arg in Args:
            _LoadJsonFile(Arg)
    elif isinstance(Args, utils_torch.PyObj):
        _LoadJsonFile(Args)
    else:
        raise Exception()

def _LoadJsonFile(Args):
    Obj = utils_torch.json.JsonFile2PyObj(Args.FilePath)
    if not Args.MountPath.startswith("&^"):
        raise Exception()
    MountPath = Args.MountPath.replace("&^", "")
    SetAttrs(ArgsGlobal, MountPath, Obj)

def LoadParamFile(Args):
    if isinstance(Args, utils_torch.PyObj):
        Args = GetAttrs(Args)
    if isinstance(Args, dict):
        _LoadParamFile(utils_torch.json.JsonObj2PyObj(Args))
    elif isinstance(Args, list):
        for Arg in Args:
            _LoadParamFile(Arg)
    elif isinstance(Args, utils_torch.PyObj):
        LoadParamFile(Args)
    else:
        raise Exception()

def _LoadParamFile(Args):
    Obj = utils_torch.json.JsonFile2PyObj(Args.FilePath)
    if not isinstance(Obj, list):
        EnsureAttrs(Args, "SetResolveBase", default=True)
        if Args.SetResolveBase:
            setattr(Obj, "__ResolveBase__", True)
    if not Args.MountPath.startswith("&^"):
        raise Exception()
    MountPath = Args.MountPath.replace("&^", "")
    SetAttrs(ArgsGlobal, MountPath, Obj)
    return

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

def ParseParamStaticAndDynamic(Args):
    ParseParamStatic(Args)
    ParseParamDynamic(Args)
    return
def ParseParamStatic(Args, Save=True, SavePath=utils.ArgsGlobal.SaveDir + "param_parsed_static.jsonc"):
    import utils_torch
    # for attr, param in utils_torch.ListAttrsAndValues(ArgsGlobal.param):
    #     utils_torch.parse.ParsePyObjStatic(param, ObjCurrent=param, ObjRoot=utils.ArgsGlobal, InPlace=True)
    param = ArgsGlobal.param
    utils_torch.parse.ParsePyObjStatic(param, ObjCurrent=param, ObjRoot=utils.ArgsGlobal, InPlace=True)
    if Save:
        SavePath = utils_torch.RenameIfPathExists(SavePath)
        utils_torch.json.PyObj2JsonFile(ArgsGlobal.param, SavePath)
    return
def ParseParamDynamic(Args, Save=True, SavePath=utils.ArgsGlobal.SaveDir + "param_parsed_dynamic.jsonc"):
    import utils_torch
    for attr, param in utils_torch.ListAttrsAndValues(ArgsGlobal.param):
        utils_torch.parse.ParsePyObjDynamic(param, ObjCurrent=param, ObjRoot=utils.ArgsGlobal, InPlace=True)
    if Save:
        utils_torch.json.PyObj2JsonFile(ArgsGlobal.param, utils_torch.RenameIfPathExists(SavePath))
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

