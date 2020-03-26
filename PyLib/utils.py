from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, errno
import logging

def list_difference(a, b):
    '''
    :param a:
    :param b:
    :return:
    '''
    set_a = set(a)
    set_b = set(b)
    comparison = set_a.difference(set_b)

    return list(comparison)

def mkdir_p(path):
    '''
    :param path:
    :return:
    '''
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def makefolders(subfolders):
    '''
    create multiple folders
    :param subfolders:
    :return:
    '''
    assert isinstance(subfolders, list)

    for path in subfolders:
        if not os.path.exists(path):
            mkdir_p(path)

def setLogConfig():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger
