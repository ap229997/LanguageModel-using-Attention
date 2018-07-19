# Functions to manage dictionaries 

import sys
import pickle
import gzip
import json
import collections

'''
Save the dictionary in .pkl format

Arguments:
    file_path : specify the path where to dump the file
    gz : whether to compress it or not

Returns:
    None
'''
def pickle_dump(data, file_path, gz=False):
    open_fct = open
    if gz:
        open_fct = gzip.open
        file_path += ".gz"

    with open_fct(file_path, "wb") as f:
        pickle.dump(data, f)

'''
Load the dictionary from .pkl format

Arguments:
    file_path : specify the path from where to load the dictionary
    gz : whether the file is in compressed form or not

Returns:
    loaded dictionary
'''
def pickle_loader(file_path, gz=False):
    open_fct = open
    if gz:
        open_fct = gzip.open

    with open_fct(file_path, "rb") as f:
        if sys.version_info > (3, 0):  # Workaround to load pickle data python2 -> python3
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            return u.load()
        else:
            return pickle.load(f)

'''
Flatten the specified input

Arguments:
    x : input to be flattened

Returns:
    flattened object in list format
'''
def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

'''
Saving the dictionary in .json format

Arguments:
    file_path : specify the path where to save the .json file
    json_string : dictionary in .json format which is to be saved

Returns:
    None
'''
def dump_json(file_path, json_string):
    with open(file_path, "wb") as f:
        results_json = json.dumps(json_string)
        f.write(results_json.encode('utf8', 'replace'))




