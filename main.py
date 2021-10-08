
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
        TaskObj = utils_torch.LoadTaskFile()
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

def CleanLog():
    import utils_torch
    utils_torch.files.RemoveAllFilesAndDirs("./log/")

def CleanFigures():
    import utils_torch
    utils_torch.files.RemoveMatchedFiles("./", r".*\.png")

if __name__=="__main__":
    main()

