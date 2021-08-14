import json
import json5
import time
import warnings
import logging
import os
import traceback

args_global = {
    "config_dicts":[

    ]
}

def main():
    set_logger_global()
    logger = get_logger_global()
    try: # catch all unhandled exceptions
        with open('./task.jsonc') as f:
            tasks = json5.load(f)
        for task in tasks:
            logger.debug("main: doing task: %s"%task['name'])
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
    file_handler = logging.FileHandler("./log/%s-%s.txt"%(logger_name, get_time()), mode='w', encoding='utf-8')  # 不拆分日志文件，a指追加模式,w为覆盖模式
    file_handler.setLevel(logging.DEBUG)            
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

def add_log(logger, log, time_stamp=True):
    if time_stamp:
        logger.log("[%s]%s"%(get_time(), log))
    else:
        logger.log("%s"%log)

def add_config_dict(args):
    file_path = args["path"]
    with open(file_path, 'r') as f:
        config_dict = json.load(f)
    args_global["config_dicts"].append(add_config_dict)
    return

def get_time(verbose=False):
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) # Time display style: 2016-03-20 11:45:39
    if verbose:
        print(time_str)
    return time_str

def get_config_sys_from_json():
    with open('./config_sys.jsonc', 'r') as f:
        config_sys = json.load(f)
    #systype = config_sys['os.type']
    #print(systype)
    return config_sys

def build_model(args):
    model_type = args['name']
    
    if model_type in ['rnn']:
        build_rnn(args)
def build_rnn(args):
    from Models import rnn
    return rnn(args)

def add_config_file(args):
    file_path = args['path']
    
    
    return
    
def add_lib(args):
    lib_name = args['name']
    lib_path = args['path']
    success = False
    if lib_path=="!get_from_config":
        for config_dict in args_global["config_dicts"]:
            if config_dict.get("libs") is not None:
                libs = config_dict["libs"]
                if libs.get(lib_name) is not None:
                    lib_path = libs.get(lib_name)
                    break
        if success:
            sys.path.append(lib_path)
        else:
            warnings.warn('add_lib failed: cannot find path to lib %s'%lib_name)
    else:
        warnings.warn('add_lib: invalid lib_path: ', lib_path)
task_implement_method = {
    'build_model': build_model,
    "add_lib": add_lib,
    "add_config_dict": add_config_dict
}

if __name__=="__main__":
    main()