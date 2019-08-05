import configparser
import ast

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def is_bool(s):
    if s == 'True' or s == 'False':
        return True
    return False

def is_none(s):
    if s == 'None':
        return True
    return False

def load_configfiles():
    config = configparser.ConfigParser()
    config.read('./configfiles/template_config.ini')
    configList = {}
    for topic in config:
        for key in config[topic]:
            configList.update({key: config[topic][key]})
    
    # change type
    configList['class_list'] = ast.literal_eval(configList['class_list'])
    # change lr to float
    if is_float(configList['lr']):
        configList['lr'] = float(configList['lr'])
    # change another
    for element in configList:
        if element == 'class_list' or element == 'lr':
            continue
        if is_int(configList[element]):
            configList[element] = int(configList[element])
            continue
        if (is_bool(configList[element])):
            configList[element] = bool(configList[element] == 'True')
            continue
        if (is_none(configList[element])):
            configList[element] = None
            continue

    # checking
    for element in configList:
        print(element, "=", configList[element], "- type =", type(configList[element]))
    
    return configList