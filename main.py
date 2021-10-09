from re import L
import sys
import argparse
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("task", nargs="?", default="DoTasksFromFile")
parser.add_argument("-IsDebug", default=True)
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
import utils
utils.Init()

import utils_torch
from utils_torch.attrs import *
utils_torch.SetArgsGlobal(utils.ArgsGlobal)

def CleanLog():
    utils_torch.files.RemoveAllFilesAndDirs("./log/")

def CleanFigures():
    utils_torch.files.RemoveMatchedFiles("./", r".*\.png")

if __name__=="__main__":
    main()