# Dependencies import
import glob
import pandas as pd
from pandas.errors import EmptyDataError

import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import medfilt
from hampel import hampel


def files_names(directory_list: list, directory_name: str):
    """
    Getting a list of files from a directory
    :return: two list. The first one is a list with the names of the files, the second one is a list of files
    """

    # Getting the files list from directories
    list_index = directory_list.index(directory_name)
    file_list = glob.glob(directory_list[list_index] + '\*.txt', recursive=False)

    tmp = []
    only_names = []

    # Removing dots and slashes from reading file format
    for fnam in file_list:
        tmp.append(fnam.rsplit('.', 1)[0])
        only_names.append(fnam.rsplit('\\', 1)[1])

    return only_names, file_list


# Se itera sobre la variable de archivos experimentales y se separan los datos
def read_multiple_files(file_list: list,
                        delimiter: str = ' ',
                        encoding: str = 'UTF-16LE',
                        skiprows=14,
                        header=None,
                        ):
    """
    Read multiple files from a diretory.
    :param file_list: list().
    :param delimiter: str, default ' ' (white space), equivalent to setting '\s+'. Delimiter to use.
    :param encoding: str, optional, default = "UTF-16LE".
    :param skiprows: list-like, int or callable, default = 14.
    :param header: int, list of int, None, default ‘infer’, default = None.
    Row number(s) to use as the column names, and the start of the data.
    :return:
    """
    # to save list of data frames to process
    good_files = []
    # to save list of files with errors
    bad_files = []

    for one_file in file_list:
        print(f"Analyzing: {one_file}")

        try:
            df = pd.read_table(one_file,
                               sep=delimiter,
                               encoding=encoding,
                               skiprows=skiprows,
                               header=header)
            head = ["idx", "t", "Th", "Tm", "TG", "f"]
            df.columns = head
            df.drop('TG', inplace=True, axis=1)
            df.drop('idx', inplace=True, axis=1)
            df.drop('Th', inplace=True, axis=1)
            good_files.append(df)

        except EmptyDataError:
            print(f"No columns to parse from file {one_file}")
            bad_files.append(one_file)
    print('Done analyzing files.')

    return good_files, bad_files
