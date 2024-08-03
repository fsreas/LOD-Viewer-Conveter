# +D 2023.11.2
# Last modified time: 2023.11.9
import yaml
import re
import numpy as np
from easydict import EasyDict as edict
from argparse import ArgumentParser, Namespace

def read_yaml(fname):
    with open(fname) as file:
        opt = edict(yaml.safe_load(file))
    return opt

def string_to_namespace(input_string):
    # Remove the 'Namespace' part from the string
    input_string = input_string.replace(
                        'Namespace(', '{').replace(')', '}')

    # Replace '=' with ':'
    input_string = input_string.replace('=', ':')
    
    # Wrap keys in quotes
    input_string = re.sub(r'(\w+):', r'"\1":', input_string)
        # NOTE: Copilot auto-completes the above line
        # NOTE: It makes the strings as the variables
    
    # Convert the string to a dictionary
    input_dict = eval(input_string)

    # Convert the dictionary to a Namespace object
    namespace = Namespace(**input_dict)

    return namespace

def read_cam_cfg(fname):
    # read txt file to one string   
    with open(fname, 'r') as file:
        content = file.read()
    return string_to_namespace(content)