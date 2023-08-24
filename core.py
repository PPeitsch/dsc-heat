# Dependencies import
import glob
from datetime import time

import pandas as pd
from pandas.errors import EmptyDataError
import matplotlib.pyplot as plt

import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import medfilt
from hampel import hampel


def tellme(s):
    print(s)
    plt.title(s, fontsize=16)   # Cambiar por una ventana, pop up message??
    plt.draw()                  # Cambiar por una ventana, pop up message??


def point_selection():
    tellme('You will select two points of the plot, click to begin')
    plt.waitforbuttonpress()

    while True:
        pts = []
        while len(pts) < 2:
            tellme('Select 2 points with the right clic of the mouse')
            pts = np.asarray(plt.ginput(2, timeout=-1))
            if len(pts) < 2:
                tellme('Too few points, starting over')
                time.sleep(1)  # Wait a second

        ph = plt.fill(pts[:, 0], pts[:, 1], 'r', lw=2)

        tellme('Happy? Key click for yes, mouse click for no')

        if plt.waitforbuttonpress():
            break

        # Get rid of fill
        for p in ph:
            p.remove()


def onclick(event):
    pos = [[event.xdata, event.ydata]]


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
        tmp = fnam.rsplit('.', 1)[0]
        only_names.append(tmp.rsplit('\\', 1)[1])

    return only_names, file_list


# Se itera sobre la variable de archivos experimentales y se separan los datos
def read_multiple_files(file_list: list,
                        delimiter: str = '\s+',
                        encoding: str = 'UTF-16LE',
                        skiprows=13,
                        header=None,
                        ):
    """
    Read multiple files from a diretory.
    :param file_list: list().
    :param delimiter: str, default '\s+' (white space). Delimiter to use.
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
            # df.drop(['TG', 'idx', 'Th'], inplace=True, axis=1)
            good_files.append(df)

        except EmptyDataError:
            print(f"No columns to parse from file {one_file}")
            bad_files.append(one_file)
    print('Done analyzing files.\n')

    return good_files, bad_files


def graph_selection(files_to_process: list,
                    directory_name: str,
                    file_name: list,
                    figure_size: tuple,
                    to_graph: tuple,
                    x_label: str,
                    y_label: str,
                    show_plot: bool = False,
                    plot_cut: bool = False,
                    face_color: str = "#6D9EC1",
                    line_style: str = '-',
                    line_width: float = 0.5,
                    line_color: str = 'white',
                    ):
    """
    This is a function to interactively plot.

    :param files_to_process: list of files to process
    :param directory_name: directory on the current path
    :param file_name: file name
    :param figure_size: tuple with plot size (in) (x, y)
    :param to_graph: selection of variables to graph
    :param x_label: x label to add in the plot
    :param y_label: y label to add in the plot
    :param show_plot: boolean to show or not the plot, default is False to show the plot
    :param plot_cut: boolean to indicate if you want cutting a section of the plot
    :param face_color: face color, default is #6D9EC1
    :param line_style: line style, default is '-'
    :param line_width: line width, default is 0.5
    :param line_color: line color, default is 'white'
    :return:
    """
    for file_to_graph, f_name in zip(files_to_process, file_name):
        print(f"Graphing {y_label} vs {x_label}: {f_name}")
        fig = plt.figure(figsize=figure_size)
        fig.patch.set_facecolor(face_color)  ## por que hay dos set_facecolor??
        fig.patch.set_alpha(0.15)  ## patch??

        ax = fig.add_subplot()
        ax.patch.set_facecolor(face_color)

        ax.plot(file_to_graph[to_graph[1]],
                file_to_graph[to_graph[0]],
                linestyle=line_style,
                linewidth=line_width,
                color=line_color,
                label=y_label)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.legend()

        if plot_cut:
            point_selection()

        if show_plot:
            plt.show()

        fig.savefig(f'{directory_name}\{f_name}_{y_label}_vs_{x_label}.png')
        plt.close()
    print('Done graphing.')
