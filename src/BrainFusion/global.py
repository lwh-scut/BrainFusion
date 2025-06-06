# -*- coding: utf-8 -*-
# @Time    : 2024/6/19 15:24
# @Author  : XXX
# @Site    : 
# @File    : global.py
# @Software: PyCharm 
# @Comment :
from PyQt5.QtCore import QObject, pyqtSignal


class SignalEmitter(QObject):
    finished_signal = pyqtSignal(str)


global_signal_emitter = SignalEmitter()
