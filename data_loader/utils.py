import os.path as osp
import os
import errno
def makedir_exist_ok(dirpath):
    """
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise