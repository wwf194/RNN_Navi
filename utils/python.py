def IsLegalPyName(name):
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

def CheckIsLegalPyName(name):
    if not IsLegalPyName(name):
        raise Exception("%s is not a legal python name."%name)
