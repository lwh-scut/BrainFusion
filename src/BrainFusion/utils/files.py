# -*- coding: utf-8 -*-
# @Time    : 2024/6/14 16:20
# @Author  : XXX
# @Site    : 
# @File    : files.py
# @Software: PyCharm 
# @Comment :
import os


def are_filenames_equal(file_path1, file_path2):
    # 获取文件名（包含后缀）
    filename1 = os.path.basename(file_path1)
    filename2 = os.path.basename(file_path2)

    # 去掉后缀
    name1 = os.path.splitext(filename1)[0]
    name2 = os.path.splitext(filename2)[0]

    # 比较文件名（不包括后缀）
    return name1 == name2


def getFileNameWithoutSuffix(filePath):
    fileName = os.path.basename(filePath)
    return os.path.splitext(fileName)[0]


def getFileName(filePath):
    return os.path.basename(filePath)


def compareFileSizes(file1_path, file2_path):
    size1 = os.path.getsize(file1_path)
    size2 = os.path.getsize(file2_path)

    if size1 > size2:
        return file1_path
    elif size2 > size1:
        return file2_path
    else:
        return "Both files have the same size."
