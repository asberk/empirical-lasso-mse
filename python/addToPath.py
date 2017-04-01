
def spgl1():
    import sys
    if sys.prefix is 'darwin':
        sys.path.append('../../SPGL1_python_port/')
    else:
        sys.path.append('../../')
    return
