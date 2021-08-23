
import json
import json5
import utils

class PyObjFromJson(object):
    def __init__(self, param=None):
        if param is not None:
            if type(param) is dict:
                self.from_dict(param)
            elif type(param) is list:
                self.from_list(param)
            elif type(param) is str:
                param = json.loads(param)
            else:
                raise Exception()
    def from_list(self, list):
        for index, item in enumerate(list):
            if type(item) is dict:
                list[index] = PyObjFromJson(item)
            elif type(item) is list:
                self.from_list(item)
            else:
                pass
        return list
    def from_dict(self, dict_):
        self.__dict__ = {}
        for key, value in dict_.items():
            if "." in key:
                keys = key.split(".")
                checkIsLegalPyName(key[0])
                for index in range(len(keys)):
                    key = keys[index]
                    obj = self
                    if hasattr(obj, key):
                        obj = getattr(self, key)
                    else:
                        if index == len(keys) - 1:
                            setattr(obj, key, PyObjFromJson({
                                keys[index]: value
                            }))
                        else:
                            setattr(obj, key, PyObjFromJson({
                                ".".join(keys[index + 1:]): value
                            }))
            else:
                checkIsLegalPyName(key)
                if type(value) is dict:
                    value = PyObjFromJson(value)
                elif type(value) is list:
                    self.from_list(value)
                else:
                    pass
                setattr(self, key, value)
        return self
    def to_dict(self):
        d = {}
        for key, value in self.__dict__.items():
            if type(value) is PyObjFromJson:
                value = value.to_dict()
            d[key] = value
        return d
    def __repr__(self):
        return str(self.to_dict())
    def __setitem__(self, key, value):
        self.__dict__[key] = value
    def __getitem__(self, key):
        return self.__dict__[key]

def JsonObj2PyObj(json_obj):
    if isinstance(json_obj, list):
        return PyObjFromJson().from_list(json_obj)
    elif isinstance(json_obj, dict):
        return PyObjFromJson().from_dict(json_obj)
    else:
        raise Exception()
json_obj_to_object = JsonObj2PyObj
JsonObj2PyObj = JsonObj2PyObj

def JsonObj2JsonStr(json_obj):
    return json.dumps(json_obj)

json_obj_to_json_str = JsonObj2JsonStr

def PyObj2JsonObj(obj):
    return json.loads(object_to_json_str(obj))

def PyObj2JsonFile(obj, path):
    # json_obj = PyObj2JsonObj(obj)
    # json_str = JsonObj2JsonStr(json_obj)
    json_str = PyObj2JsonStr(obj)
    JsonStr2JsonFile(json_str, path)

def JsonStr2JsonFile(json_str, path):    
    if path.endswith(".jsonc") or path.endswith(".json"):
        pass
    else:
        path += ".jsnonc"
    with open(path, "w") as f:
        f.write(json_str)

object_to_json_obj = PyObj2JsonObj

def PyObj2JsonStr(obj):
    # return json.dumps(obj.__dict__, cls=change_type,indent=4)
    # why default=lambda o: o.__dict__?
    return json.dumps(obj, default=lambda o: o.__dict__, sort_keys=False, indent=4)
object_to_json_str = PyObj2JsonStr

def JsonStr2JsonObj(json_str):
    return json.loads(json_str)
json_str_to_json_obj = JsonStr2JsonObj

def JsonStr2PyObj(json_str):
    json_obj = json_str_to_json_obj(json_str)
    return json_obj_to_object(json_obj)
    # return json.loads(json_str, object_hook=lambda d: SimpleNamespace(**d))
json_str_to_object = JsonStr2PyObj

def JsonFile2PyObj(file_path):
    return JsonObj2PyObj(JsonFile2JsonObj(file_path))

def JsonFile2JsonObj(file_path):
    with open(file_path, "r") as f:
        json_dict = json5.load(f)
    return json_dict
load_json_file = JsonFile2JsonObj

def JsonStr2JsonFile(json_str, path):
    with open(path, "w") as f:
        f.write(json_str)

new_json_file = JsonStr2JsonFile

def parse_param_json_obj(json_obj, overwrite=True):
    py_obj = JsonObj2PyObj(json_obj)
    for name, obj in utils_torch.list_attrs(py_obj):
        setattr(obj, "__DollarPath__", "root.%s."%name)
    json_dicts_parsed = parse_py_obj(py_obj)
    for value in json_dicts_parsed.values():
        value.pop("__DollarPath__")
    for name, obj in utils_torch.list_attrs(py_obj):
        delattr(obj, "__DollarPath__")
    return json_dicts_parsed

def parse_param_py_obj(py_obj, overwrite=True):
    for name, obj in utils_torch.list_attrs(py_obj):
        setattr(obj, "__DollarPath__", "root.%s."%name)
    py_obj_parsed = parse_py_obj(py_obj)
    for name, obj in utils_torch.list_attrs(py_obj_parsed):
        delattr(obj, "__DollarPath__")
    return py_obj_parsed

def parse_json_obj(json_obj): # obj can either be dict or list.
    py_obj = JsonObj2PyObj(json_obj)
    parse_py_obj(py_obj)
    return py_obj

JsonObj2ParsedJsonObj = parse_json_obj

def JsonObj2ParsedPyObj(json_obj):
    return JsonObj2PyObj(JsonObj2ParsedJsonObj(json_obj))

def parse_py_obj(py_obj):
    _parse_py_obj(py_obj, root=py_obj, attrs=[], parent=None)
    return py_obj

def _parse_py_obj(obj, root, attrs, parent, base_path="root."):
    if hasattr(obj, "__DollarPath__"):
        base_path = getattr(obj, "__DollarPath__")
    if isinstance(obj, list):
        for index, item in enumerate(obj):
            _parse_py_obj(item, root, attrs + [index], obj, base_path)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            _parse_py_obj(value, root, attrs + [key], obj, base_path)
    elif isinstance(obj, str):
        sentence = obj
        while type(sentence) is str and ("@" in sentence or "^" in sentence):
            sentence = sentence.replace("@", base_path).replace("^", "root.")
            sentence = eval(sentence)
        if isinstance(sentence, str) and sentence.startswith("$"):
            sentence = eval(sentence[1:])
        
        parent[attrs[-1]] = sentence
    elif isinstance(obj, object) and hasattr(obj, "__dict__"):
        for attr, value in obj.__dict__.items():
            _parse_py_obj(value, root, attrs + [attr], obj, base_path)
    else:
        #raise Exception("_parse_json_obj: Invalid type: %s"%type(obj))
        pass
    return obj

def isLegalPyName(name):
    if name=="":
        return False
    if name[0].isalpha() or name[0] == '_':
        for i in name[1:]:
            if not (i.isalnum() or i == '_'):
                return False
        else:
            return True
    else:
        return False

def checkIsLegalPyName(name):
    if not isLegalPyName(name):
        raise Exception("%s is not a legal python name."%name)