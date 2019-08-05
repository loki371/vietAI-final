import configparser
import ast

def load_configfiles():
    config = configparser.ConfigParser()
    config.read('./configfiles/template_config.ini')
    configList = {}
    for topic in config:
        for key in config[topic]:
            configList.update({key: config[topic][key]})
    configList['class_list'] = ast.literal_eval(configList['class_list'])
    return configList