# -*- coding: utf-8 -*-
# python 3.x
# Filename:const.py

class _const:
    class ConstError(TypeError): pass
    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("Can't rebind const (%s)" % name)
        self.__dict__[name] = value
import sys
sys.modules[__name__] = _const()
