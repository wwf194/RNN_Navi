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

def CheckIsPyObj(Obj):
    if not IsPyObj(Obj):
        raise Exception("Obj is not PyObj, but %s"%type(Obj))

def IsPyObj(Obj):
    return isinstance(Obj, PyObj)

def IsDictLikePyObj(Obj):
    return isinstance(Obj, PyObj) and not Obj.IsListLike()

def IsListLikePyObj(Obj):
    return isinstance(Obj, PyObj) and Obj.IsListLike()

class PyObjCache(object):
    def __init__(self):
        return

class PyObj(object):
    def __init__(self, param=None, data=None, **kw):
        self.cache = PyObjCache()
        if param is not None:
            if type(param) is dict:
                self.FromDict(param)
            elif type(param) is list:
                self.FromList(param)
            else:
                raise Exception()
    def __repr__(self):
        return str(self.ToDict())
    def __setitem__(self, key, value):
        if hasattr(self, "__value__") and isinstance(self.__value__, list):
            self.__value__[key] = value
        else:
            self.__dict__[key] = value
    def __getitem__(self, index):
        if hasattr(self, "__value__") and isinstance(self.__value__, list):
            return self.__value__[index]
        else:
            return self.__dict__[index]
    def __len__(self):
        return len(self.__value__)
    def __str__(self):
        return utils.json.PyObj2JsonStr(self)
    def FromList(self, List):
        ListParsed = []
        for Index, Item in enumerate(List):
            if type(Item) is dict:
                ListParsed.append(PyObj(Item))
            elif type(Item) is list:
                ListParsed.append(PyObj(Item))
            else:
                ListParsed.append(Item)
        self.__value__ = ListParsed
        return self
    def FromDict(self, Dict):
        #self.__dict__ = {}
        for key, value in Dict.items():
            # if key in ["Init.Method"]:
            #     print("aaa")
            if key in ["__ResolveRef__"]:
                if not hasattr(self, "__ResolveRef__"):
                    setattr(self.cache, key, value)
                continue

            if "." in key:
                keys = key.split(".")
            else:
                keys = [key]
            utils.python.CheckIsLegalPyName(key[0])
            obj = self
            parent, parentAttr = None, None
            for index, key in enumerate(keys):
                if index == len(keys) - 1:
                    if hasattr(obj, key):
                        valueOld = getattr(obj, key) 
                        if isinstance(valueOld, PyObj) and valueOld.IsDictLike():
                            if isinstance(value, PyObj) and value.IsDictLike():
                                valueOld.FromPyObj(value)
                            elif isinstance(value, dict):
                                valueOld.FromDict(value)
                            else:
                                # if hasattr(value, "__value__"):
                                #     utils_torch.AddWarning("PyObj: Overwriting key: %s. Original Value: %s, New Value: %s"\
                                #         %(key, getattr(obj, key), value))                                       
                                setattr(valueOld, "__value__", value)
                        else:
                            # utils_torch.AddWarning("PyObj: Overwriting key: %s. Original Value: %s, New Value: %s"\
                            #     %(key, valueOld, value))
                            setattr(obj, key, self.ProcessValue(key, value))
                    else:
                        if isinstance(obj, PyObj):
                            setattr(obj, key, self.ProcessValue(key, value))
                        else:
                            setattr(parent, parentAttr, PyObj({
                                "__value__": obj,
                                key: self.ProcessValue(key, value)
                            }))
                else:
                    if hasattr(obj, key):
                        parent, parentAttr = obj, key
                        obj = getattr(obj, key)
                    else:
                        if isinstance(obj, PyObj):
                            setattr(obj, key, PyObj())
                            parent, parentAttr = obj, key
                            obj = getattr(obj, key)
                        else:
                            setattr(parent, parentAttr, PyObj({
                                "__value__": obj,
                                key: PyObj()
                            }))
                            obj = getattr(parent, parentAttr)
                            parent, parentAttr = obj, key
                            obj = getattr(obj, key)
        return self
    def FromPyObj(self, Obj):
        self.FromDict(Obj.__dict__)
    def ProcessValue(self, key, value):
        if isinstance(value, dict):
            return PyObj(value)
        elif isinstance(value, list):
            if key in ["__value__"]:
                return value
            else:
                return PyObj(value)
        elif type(value) is PyObj:
            return value
        else:
            return value
    def GetList(self):
        if not self.IsListLike():
            raise Exception()
        return self.__value__
    def IsListLike(self):
        return hasattr(self, "__value__") and isinstance(self.__value__, list)
    def IsDictLike(self):
        return not self.IsListLike()
    def SetResolveBase(self):
        self.__ResolveBase__ = True
    def IsResolveBase(self):
        if hasattr(self, "__ResolveBase__"):
            if self.__ResolveBase__==True or self.__ResolveBase__ in ["here"]:
                return True
        return False
    def append(self, content):
        if not self.IsListLike():
            raise Exception()
        self.__value__.append(content)
    def ToDict(self):
        Dict = {}
        for key, value in ListAttrsAndValues(self, Exceptions=["__ResolveRef__"]):
            if type(value) is PyObj:
                value = value.ToDict()
            Dict[key] = value
        return Dict

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

