
def spgl1():
    import sys
    if sys.prefix is 'darwin' or sys.prefix.__contains__('berkas'):
        sys.path.append('./../../SPGL1_python_port/')
    else:
        sys.path.append('../..')
    return
