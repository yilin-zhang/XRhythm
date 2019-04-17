# -*-coding: utf-8-*-
# Author: Yilin Zhang

import os


def get_file_path(directory, suffix):
    '''Generate the paths of all the given type files in the given directory.
    Arg:
    - directory: The directory that contains the files you need.
    - suffix: The suffix of the file type.

    Return:
    - path: The file path.
    - file: The file name.
    '''
    for root, _, files in os.walk(directory):
        for file_name in files:
            current_suffix = os.path.splitext(file_name)[1]

            if current_suffix != suffix:
                continue

            path = root + "/" + file_name
            yield path, file_name
