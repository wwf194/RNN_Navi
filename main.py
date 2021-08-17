import os
import sys
import argparse
import json
import json5
import time
import warnings
import logging
import traceback

args_global = {
    "config_dicts":[]
}

def main():
    set_logger_global()
    logger = get_logger_global()
    try: # catch all unhandled exceptions
        tasks = load_json('./task.jsonc')
        for task in tasks:
            #add_log("main: doing task: %s"%task['name'])
            #print(task)
            task_implement_method[task['name']](task['args'])
    except Exception:
        logger.error(traceback.format_exc())

def set_logger_global():
    args_global["logger_global"] = get_logger('log-global')

def get_logger_global():
    return args_global["logger_global"]

def get_logger(logger_name='log'):
    # 输出到console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG) # 指定被处理的信息级别为最低级DEBUG，低于level级别的信息将被忽略
    if not os.path.exists("./log/"):
        os.mkdir("./log/")
    # 输出到file
    file_handler = logging.FileHandler("./log/%s-%s.txt"%(logger_name, get_time("%Y-%m-%d-%H:%M:%S")), mode='w', encoding='utf-8')  # 不拆分日志文件，a指追加模式,w为覆盖模式
    file_handler.setLevel(logging.DEBUG)            
    logger = logging.getLogger("LoggerMain")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

def add_log(log, logger=None, time_stamp=True):
    if logger is None:
        logger = get_logger_global()
    if time_stamp:
        logger.debug("[%s]%s"%(get_time(), log))
    else:
        logger.debug("%s"%log)

def add_warning(log, logger=None, time_stamp=True):
    if logger is None:
        logger = get_logger_global()
    if time_stamp:
        logger.warning("[%s][WARNING]%s"%(get_time(), log))
    else:
        logger.warning("%s"%log)

def add_error(log, logger=None, time_stamp=True):
    if logger is None:
        logger = get_logger_global()
    if time_stamp:
        logger.error("[%s][ERROR]%s"%(get_time(), log))
    else:
        logger.error("%s"%log)

def get_time(format="%Y-%m-%d %H:%M:%S", verbose=False):
    time_str = time.strftime(format, time.localtime()) # Time display style: 2016-03-20 11:45:39
    if verbose:
        print(time_str)
    return time_str

def get_config_sys_from_json():
    with open('./config_sys.jsonc', 'r') as f:
        config_sys = json5.loads(f)
    #systype = config_sys['os.type']
    #print(systype)
    return config_sys

def build_model(args):
    model_type = args["type"]
    add_log("Building model...type: %s"%model_type)
    if args.get("path") is not None:
        add_log("Loading model")
        args = load_json(args["path"])
    if model_type in ["rnn_lif"]:
        build_rnn_lif(args)

def build_rnn_lif(args):
    add_log("Building RNN_LIF from args.")
    import Models
    return Models.rnn_lif.init_model(args)

def load_json(file_path):
    print(file_path)
    with open(file_path, "r") as f:
        json_dict = json5.load(f)
    return json_dict

def add_config_dict(args):
    file_path = args["path"]
    with open(file_path, 'r') as f:
        config_dict = json5.load(f) # json5 allows comment in .json file.
    args_global["config_dicts"].append(config_dict)
    add_log("Added configuration dict from file: %s"%file_path, get_logger_global(), )
    return

def add_lib_path(args):
    lib_name = args['name']
    lib_path = args['path']
    if lib_path=="!get_from_config":
        success = False
        for config_dict in args_global["config_dicts"]:
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

task_implement_method = {
    'build_model': build_model,
    "add_lib_path": add_lib_path,
    "add_config_dict": add_config_dict
}

if __name__=="__main__":
    main()

