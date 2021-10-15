from re import L
import sys
import argparse
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("task", nargs="?", default="DoTasksFromFile")
parser.add_argument("-IsDebug", default=True)
parser.add_argument("-sd", "--SaveDir", dest="SaveDir", default=None)
Args = parser.parse_args()

TaskFilePath = "./task.jsonc"

def main():
    if Args.task in ["CleanLog", "CleanLog", "cleanlog"]:
        CleanLog()
    elif Args.task in ["CleanFigure"]:
        CleanFigures()
    elif Args.task in ["DoTasksFromFile"]:
        TaskObj = utils_torch.LoadTaskFile(TaskFilePath)
        TaskList = utils_torch.ParseTaskObj(TaskObj)
        if not Args.IsDebug:
            try: # catch all unhandled exceptions
                utils_torch.DoTasks(TaskList, ObjRoot=utils_torch.GetArgsGlobal())
            except Exception:
                utils_torch.AddError(traceback.format_exc())
                raise Exception()
        else:
            utils_torch.DoTasks(TaskList, ObjRoot=utils_torch.GetArgsGlobal())
    elif Args.task in ["TotalLines"]:
        utils_torch.CalculateGitProjectTotalLines()
    else:
        raise Exception("Inavlid Task: %s"%Args.task)

def ScanConfigFile(FilePath="./config.jsonc"):
    import json5
    with open(FilePath, "r") as f:
        config = json5.load(f) # json5 allows comments. config is either dict or list.
    sys.path.append(config["Library"]["utils_torch"]["IncludePath"])
    import utils_torch
    import utils
    utils_torch.attrs.SetAttrs(utils.ArgsGlobal, "config", utils_torch.PyObj(config)) # mount config on utils_torch.ArgsGlobal.config
ScanConfigFile()

def ParseMainTask(task):
    if task in ["CleanLog", "CleanLog", "cleanlog"]:
        task = "CleanLog"
    elif task in ["DoTasksFromFile"]:
        task = "DoTasksFromFile"
    else:
        raise Exception(task)
    return task

def InitUtils():
    import utils_torch
    import utils
    Args.task = ParseMainTask(Args.task)
    utils_torch.SetArgsGlobal(ArgsGlobal=utils.ArgsGlobal)
    if Args.SaveDir is not None: # Create
        utils_torch.SetSaveDir(ArgsGlobal=utils.ArgsGlobal, Name=utils.SaveDirName)
    else:
        utils_torch.SetSaveDir(ArgsGlobal=utils.ArgsGlobal, Name=Args.task)
    utils_torch.SetLoggerGlobal(ArgsGlobal=utils.ArgsGlobal)
InitUtils()

import utils
import utils_torch
from utils_torch.attrs import *
utils_torch.SetArgsGlobal(utils.ArgsGlobal)

def CleanLog():
    utils_torch.files.RemoveAllFilesAndDirs("./log/")

def CleanFigures():
    utils_torch.files.RemoveMatchedFiles("./", r".*\.png")

if __name__=="__main__":
    main()