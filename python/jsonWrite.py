import json
import codecs
import numpy as np
import sys


def dict(filename, obj):
    """
    Take a dictionary of np.ndarray objects (and possibly other types of
    objects) and write it to a file in json format (utf-8 encoding).
    Input:
    filename: filepath (including extension) of file to create/write to
    obj: the dictionary object to be written to output. Untested for types
    other than dictionary of np.ndarray objects.
    """
    for key, value in obj.items():
        if isinstance(obj[key], np.ndarray):
            obj[key] = value.tolist()
    try:
        with codecs.open(filename, 'w', encoding='utf-8') as fp:
            json.dump(obj, fp, separators=(',', ':'), indent=4, sort_keys=True)
    except Exception as e:
        print(e)
        print("Unexpected error:", sys.exc_info()[0])
        raise


def array(filename, arr):
    if isinstance(arr, np.ndarray):
        arr = arr.tolist()
    try:
        with codecs.open(filename, 'a', encoding='utf-8') as fp:
            json.dump(arr, fp, separators=(',', ':'), indent=4)
    except Exception as e:
        print(e)
        print('Unexpected error: ', sys.exc_info()[0])
        raise
