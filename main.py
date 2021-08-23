import os
import sys
import argparse
import json
import json5
import time
import warnings
import logging
import traceback

sys.path.append("./utils/")
from utils import *

def main():
    set_logger_global()
    logger = get_logger_global()
    try: # catch all unhandled exceptions
        tasks = JsonFile2JsonObj('./task.jsonc')
        for task in tasks:
            #add_log("main: doing task: %s"%task['name'])
            #print(task)
            TaskImplementFunction[task['name']](task['args'])
    except Exception:
        logger.error(traceback.format_exc())

def build_object(args):
    if isinstance(args, list):
        for arg in args:
            _build_object(arg)
    elif isinstance(args, dict):
        _build_object(args)

def _build_object(args):
    import utils_torch
    name = args["name"]
    Class = utils_torch.import_module(args["ModulePath"])
    args_global["objects"][name] = Class.init_from_param(utils_torch.JsonObj2PyObj(args_global["param_dicts"][args["ParameterName"]]))

def build_model(args):
    import utils_torch
    method = args.setdefault("method", "FromParameter")
    if method in ["FromParameter"]:
        args_model = args_global["param_dicts"][args["ParameterName"]]
    elif method in ["FromPath"]:
        args_model = JsonFile2JsonObj(args["path"])
    else:
        raise Exception()
    param = utils_torch.JsonObj2PythonObj(args_model)
    if param.type in ["rnn_lif"]:
        add_log("Building RNN_LIF from args.")
        import Models
        return Models.rnn_lif.init_model(param)
    else:
        raise Exception()

def JsonFile2JsonObj(file_path):
    with open(file_path, "r") as f:
        json_dict = json5.load(f)
    return json_dict

def load_parameter_file(args):
    if isinstance(args, dict):
        _load_parameter_file(args)
    elif isinstance(args, list):
        for args_dict in args:
            _load_parameter_file(args_dict)
    else:
        raise Exception()

def _load_parameter_file(args):
    path, name = args["path"], args["name"]
    args_global["param_dicts"][name] = JsonFile2JsonObj(path)
    add_log("Loading parameter file %s to parameter %s"%(path, name))

def load_configuration_file(args):
    if isinstance(args, dict):
        _load_configuration_file(args)
    elif isinstance(args, list):
        for args_dict in args:
            _load_configuration_file(args)
    else:
        raise Exception()

def _load_configuration_file(args):
    path, name = args["path"], args["name"]
    with open(path, 'r') as f:
        config_dict = json5.load(f) # json5 allows comment in .json file.
    args_global["config_dicts"][name] = JsonFile2JsonObj(path)
    add_log("Loaded configuration file from to %s config %s."%(path, name))

def add_library_path(args):
    if isinstance(args, dict):
        _add_library_path(args)
    elif isinstance(args,list):
        for args_dict in args:
            _add_library_path(args_dict)
    else:
        raise Exception()

def _add_library_path(args):
    # requires args to be a dict.
    lib_name = args['name']
    lib_path = args['path']
    if lib_path=="!get_from_config":
        success = False
        for config_name, config_dict in args_global["config_dicts"].items():
            if config_dict.get("libs") is not None:
                libs = config_dict["libs"]
                if libs.get(lib_name) is not None:
                    lib_path = libs[lib_name]["path"]
                    success = True
                    break
        if not success:
            add_warning('add_lib failed: cannot find path to lib %s'%lib_name)
            return
    if os.path.exists(lib_path):
        if os.path.isdir(lib_path):
            sys.path.append(lib_path)
            add_log("Added library <%s> from path %s"%(lib_name, lib_path))
        else:
            add_warning('add_lib failed: path %s exists but is not a directory.'%lib_path)
    else:
        add_warning('add_lib: invalid lib_path: ', lib_path)

def parse_parameter(args):
    import utils_torch
    param_json_dicts_parsed = utils_torch.utils.parse_param_json_dicts(args_global["param_dicts"])
    args_global["param_dicts_origin"] = args_global["param_dicts"]
    args_global["param_dicts"] = args_global["param_dicts_parsed"] = param_json_dicts_parsed

TaskImplementFunction = {
    "BuildModel": build_model,
    "AddLibraryPath": add_library_path,
    "LoadConfigurationFile": load_configuration_file,
    "LoadParameterFile": load_parameter_file,
    "ParseParameter": parse_parameter,
    "BuildObject": build_object,
}

if __name__=="__main__":
    main()

