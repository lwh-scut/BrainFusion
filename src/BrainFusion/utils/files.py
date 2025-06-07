# -*- coding: utf-8 -*-
# @Time    : 2024/6/14 16:20
# @Author  : XXX
# @Site    : 
# @File    : files.py
# @Software: PyCharm 
# @Comment :
import os


def are_filenames_equal(file_path1, file_path2):
    """
    Compare file names without extensions.

    :param file_path1: First file path
    :type file_path1: str
    :param file_path2: Second file path
    :type file_path2: str
    :return: True if base names match, False otherwise
    :rtype: bool
    """
    # Extract base filenames
    filename1 = os.path.basename(file_path1)
    filename2 = os.path.basename(file_path2)

    # Remove file extensions
    name1 = os.path.splitext(filename1)[0]
    name2 = os.path.splitext(filename2)[0]

    # Compare base names
    return name1 == name2


def getFileNameWithoutSuffix(filePath):
    """
    Extract filename without extension from full path.

    :param filePath: Full file path
    :type filePath: str
    :return: Base filename without extension
    :rtype: str
    """
    filename = os.path.basename(filePath)
    return os.path.splitext(filename)[0]


def getFileName(filePath):
    """
    Extract full filename from path.

    :param filePath: Full file path
    :type filePath: str
    :return: File name with extension
    :rtype: str
    """
    return os.path.basename(filePath)


def compareFileSizes(file1_path, file2_path):
    """
    Compare file sizes and return larger file path.

    :param file1_path: First file path
    :type file1_path: str
    :param file2_path: Second file path
    :type file2_path: str
    :return: Path to larger file or equality message
    :rtype: str
    """
    # Get file sizes in bytes
    size1 = os.path.getsize(file1_path)
    size2 = os.path.getsize(file2_path)

    # Compare and return result
    if size1 > size2:
        return file1_path
    elif size2 > size1:
        return file2_path
    else:
        return "Both files have the same size."