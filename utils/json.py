import re
import json
import json5
from typing import List

import utils
from utils.python import CheckIsLegalPyName
#from utils import AddWarning
# from utils_torch.utils import ListAttrs # leads to recurrent reference.

def EmptyPyObj():
    return PyObj()

class PyObj(object):
    def __init__(self, param=None):
        if param is not None:
            if type(param) is dict:
                self.FromDict(param)
            elif type(param) is list:
                self.FromList(param)
            elif type(param) is str:
                param = json.loads(param)
            else:
                raise Exception()
    def FromList(self, list):
        ListParsed = []
        for Index, Item in enumerate(list):
            if type(Item) is dict:
                ListParsed.append(PyObj(Item))
            elif type(Item) is list:
                ListParsed.append(self.FromList(Item))
            else:
                ListParsed.append(Item)
        return ListParsed
    def FromDict(self, Dict):
        #self.__dict__ = {}
        for key, value in Dict.items():
            if "." in key:
                # For debugging.
                # if "Batch.Internal" in key:
                #     print("aaa")
                keys = key.split(".")
                obj = self
                for index, key in enumerate(keys):
                    CheckIsLegalPyName(key)
                    if index == len(keys) - 1:
                        if hasattr(obj, key):
                            utils.AddWarning("PyObj: Overwriting key: %s. Original Value: %s, New Value: %s"\
                                %(key, getattr(obj, key), value))
                        #setattr(obj, key, value)
                        self.ProcessKeyValue(obj, key, value)
                    else:
                        if hasattr(obj, key):
                            obj = getattr(obj, key)
                        else:
                            # if index == len(keys) - 1:
                            #     setattr(obj, key, PyObj({
                            #         keys[index]: value
                            #     }))
                            # else:
                            #     setattr(obj, key, PyObj({
                            #         ".".join(keys[index + 1:]): value
                            #     }))
                            setattr(obj, key, PyObj())
                            obj = getattr(self, key)
            else:
                CheckIsLegalPyName(key)
                self.ProcessKeyValue(self, key, value)
        return self
    def ProcessKeyValue(self, obj, key, value):
        if type(value) is dict:
            if hasattr(obj, key) and isinstance(getattr(obj, key), PyObj):
                getattr(obj, key).FromDict(value)
            else: # overwrite
                setattr(obj, key, PyObj(value))
        elif type(value) is list:
            # always overwrite
            setattr(obj, key, obj.FromList(value))
        else:
            # alwayes overwrite
            setattr(obj, key, value)
    def to_dict(self):
        d = {}
        for key, value in self.__dict__.items():
            if type(value) is PyObj:
                value = value.to_dict()
            d[key] = value
        return d
    def __repr__(self):
        return str(self.to_dict())
    def __setitem__(self, key, value):
        self.__dict__[key] = value
    def __getitem__(self, key):
        return self.__dict__[key]

def JsonObj2PyObj(JsonObj):
    if isinstance(JsonObj, list):
        return PyObj().FromList(JsonObj)
    elif isinstance(JsonObj, dict):
        return PyObj().FromDict(JsonObj)
    else:
        raise Exception()
json_obj_to_object = JsonObj2PyObj

def JsonObj2JsonStr(json_obj):
    return json5.dumps(json_obj)

def PyObj2JsonObj(obj):
    return json.loads(PyObj2JsonStr(obj))

def PyObj2JsonFile(obj, path):
    # JsonObj = PyObj2JsonObj(obj)
    # JsonStr = JsonObj2JsonStr(JsonObj)
    JsonStr = PyObj2JsonStr(obj)
    JsonStr2JsonFile(JsonStr, path)

def JsonStr2JsonFile(JsonStr, path):    
    if path.endswith(".jsonc") or path.endswith(".json"):
        pass
    else:
        path += ".jsnonc"
    with open(path, "w") as f:
        f.write(JsonStr)

object_to_json_obj = PyObj2JsonObj

def PyObj2JsonStr(obj):
    # return json.dumps(obj.__dict__, cls=change_type,indent=4)
    # why default=lambda o: o.__dict__?
    return json.dumps(obj, default=lambda o: o.__dict__, sort_keys=False, indent=4)
    
def PyObj2JsonStrHook(Obj):
    # to be implemented
    return

def JsonStr2JsonObj(JsonStr):
    RemoveJsonStrComments(JsonStr)
    JsonStr = re.sub(r"\s", "", JsonStr)
    _JsonStr2JsonObj(JsonStr)
    #return json.loads(JsonStr)
def _JsonStr2JsonObj(JsonStr):
    #JsonStr = re.sub(r"\".*\"", "", JsonStr)
    #JsonStr = re.sub(r"\'.*\'", "", JsonStr)
    if JsonStr[0]=="{" and JsonStr[-1]=="}":
        Obj = {}
        Segments = JsonStr[1:-1].rstrip(",").split(",")
        for Segment in Segments:
            AttrValue = Segment.split(":")
            Attr = AttrValue[0]
            if not (Attr[0]=="\"" and Attr[-1]=="\"" or Attr[0]=="\'" and Attr[-1]=="\'"):
                raise Exception()
            Attr = Attr[1:-1]
            Value = AttrValue[1]
            Obj[Attr] = _JsonStr2JsonObj(Value)
        return Obj
    elif JsonStr[0]=="[" and JsonStr[-1]=="]":
        Obj = []
        Segments = JsonStr[1:-1].rstrip(",").split(",")
        for Index, Segment in enumerate(Segments):
            Obj.append(_JsonStr2JsonObj(Segment, ))    
        return Obj
    elif JsonStr[0]=="\"" and JsonStr[-1]=="\"" or JsonStr[0]=="\'" and JsonStr[-1]=="\'":
        return JsonStr[1:-1]
    try:
        Obj = int(JsonStr)
        return Obj
    except Exception:
        pass

    try:
        Obj = float(JsonStr)
        return Obj
    except Exception:
        pass
    raise Exception()

def JsonStr2PyObj(JsonStr):
    JsonObj = JsonStr2JsonObj(JsonStr)
    return JsonObj2PyObj(JsonObj)
    # return json.loads(JsonStr, object_hook=lambda d: SimpleNamespace(**d))
JsonStr_to_object = JsonStr2PyObj

def JsonFile2PyObj(FilePath):
    return JsonObj2PyObj(JsonFile2JsonObj(FilePath))

def JsonFile2JsonObj(FilePath):
    # with open(FilePath, "r") as f:
    #     JsonStrLines = f.readlines()
    # JsonStrLines = RemoveJsonStrLinesComments(JsonStrLines)
    # JsonStr = "".join(JsonStrLines)
    # JsonStr = re.sub("\s", "", JsonStr) # Remove All Empty Characters
    # JsonStr = RemoveJsonStrComments(JsonStr)
    # return JsonStr2JsonObj(JsonStr)
    with open(FilePath, "r") as f:
        JsonObj = json5.load(f) # json5 allows comments
    return JsonObj
def RemoveJsonStrLinesComments(JsonStrLines): # Remove Single Line Comments Like //...
    for Index, JsonStrLine in enumerate(JsonStrLines):
        JsonStrLines[Index] = re.sub(r"//.*\n", "", JsonStrLine)
    return JsonStrLines
def RemoveJsonStrComments(JsonStr):
    JsonStr = re.sub(r"/\*.*\*/", "", JsonStr)
    return JsonStr

load_json_file = JsonFile2JsonObj

def JsonObj2JsonFile(JsonObj, FilePath):
    return JsonStr2JsonFile(JsonObj2JsonStr(JsonObj), FilePath)

def JsonStr2JsonFile(JsonStr, path):
    with open(path, "w") as f:
        f.write(JsonStr)

new_json_file = JsonStr2JsonFile

