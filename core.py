# Dependencies import
import glob
import time

import pandas as pd
from pandas.errors import EmptyDataError
import matplotlib.pyplot as plt

import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import medfilt
from hampel import hampel

# for the yes / no window
import tkinter as tk
from tkinter import *
from tkinter import messagebox as mb

# New import for user input
from tkinter import simpledialog


def ask_user_input(prompt):
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    user_input = simpledialog.askstring("Input", prompt)
    root.destroy()  # Clean up the hidden window

    return user_input


def point_selection():
    win1 = tk.Tk()
    win1.withdraw()  # Hide the main window

    # Pause and ask if the user is ready to select points
    ready = mb.askyesno("Ready?", "Are you ready to select two points using the right mouse click?")

    win1.destroy()  # Clean up the hidden window

    if not ready:
        return

    while True:
        pts = []
        ph = None

        mb.showinfo('Select 2 points', 'Use the mouse right click to select two points.')
        pts = np.asarray(plt.ginput(2, timeout=-1))
        print(pts)
        print(ph)

        if len(pts) != 2:
            mb.showinfo('Error', 'Please select exactly two points.')
            continue

        ph = plt.fill(pts[:, 0], pts[:, 1], 'r', lw=2)
        print(pts)
        print(ph)

        win2 = tk.Tk()
        win2.withdraw()  # Hide the main window

        # Ask if the user is happy with the selection
        happy = mb.askyesno("Happy?", "Are you happy with the selected points?")

        win2.destroy()  # Clean up the hidden window

        if happy:
            print("Selected points:", pts)
            break

    return pts


def files_names(directory_list: list, directory_name: str):
    """
    Getting a list of files from a directory
    :return: two list. The first one is a list with the names of the files, the second one is a list of files
    """

    # Getting the files list from directories
    list_index = directory_list.index(directory_name)
    file_list = glob.glob(directory_list[list_index] + '\*.txt', recursive=False)

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
                    plot_baseline: bool = False,
                    baseline_data: list = None,
                    baseline_color: str = 'red',
                    baseline_style: str = 'dotted',
                    baseline_width: float = 1.0,
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
    :param plot_cut: boolean to indicate a cutoff section of the plot
    :param face_color: face color, default is #6D9EC1
    :param line_style: line style, default is '-'
    :param line_width: line width, default is 0.5
    :param line_color: line color, default is 'white'
    :param plot_baseline: boolean to indicate if baseline should be plotted
    :param baseline_data: data for the baseline plot
    :param baseline_color: color of the baseline, default is 'red'
    :param baseline_style: style of the baseline, default is 'dotted'
    :param baseline_width: width of the baseline, default is 1.0
    :return:
    """

    data_after_cut = []

    for file_to_graph, f_name in zip(files_to_process, file_name):
        print(f"Graphing {y_label} vs {x_label}: {f_name}")
        fig = plt.figure(figsize=figure_size)
        fig.patch.set_facecolor(face_color)  # por que hay dos set_facecolor??
        fig.patch.set_alpha(0.15)  # patch??

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
            points = point_selection()
            if points.all() is not None and len(points) == 2:
                left_point, right_point = sorted(points, key=lambda p: p[0])

                # Filter data between the two points
                mask = (file_to_graph[to_graph[1]] >= left_point[0]) & (file_to_graph[to_graph[1]] <= right_point[0])
                filtered_data = file_to_graph[mask]

                # Plot the filtered data
                plt.plot(filtered_data[to_graph[1]],
                         filtered_data[to_graph[0]],
                         label='Filtered Data')

                data_after_cut.append(filtered_data)

        elif show_plot:
            plt.show()

        if plot_baseline and baseline_data is not None:
            for baseline_curve_data in baseline_data:
                baseline_x = baseline_curve_data[0]
                baseline_y = baseline_curve_data[1]
                plt.plot(baseline_x, baseline_y,
                         linestyle=baseline_style,
                         linewidth=baseline_width,
                         color=baseline_color,
                         label='Baseline')

        fig.savefig(f'{directory_name}\{f_name}_{y_label}_vs_{x_label}.png')
        plt.close()
    print('Done graphing.')
    return data_after_cut


def generate_baseline(curves: list):
    """
    Generate baseline curves for given peaks and points.

    :param curves: List of curves, where each curve is represented as a 2D numpy array
                   with shape (n_points, 2) containing x and y values.
    :return: List of baseline curves corresponding to each input curve.
    """
    baseline_curves = []

    for curve in curves:
        x_values = curve.iloc[:, 0]
        y_values = curve.iloc[:, 1]

        # baseline calculation
        baseline_curve = calculate_baseline(x_values, y_values)
        baseline_curves.append([x_values, baseline_curve])

    return baseline_curves


def calculate_baseline(x_values, y_values):
    """
    Calculate the baseline for a given curve using a equation proposed by U. Bandara,
    "A SYSTEMATIC SOLUTION TO THE PROBLEM OF SAMPLE BACKGROUND CORRECTION IN DSC CURVES",
    Journal of Thermal Analysis, Vol. 31 (1986) 1063-1071

    :param x_values: Array of x values.
    :param y_values: Array of y values.
    :return: Array of baseline y values.
    """
    baseline = y_values - 0.1 * x_values

    ## Interpolación desde el punto Amin al punto de onset, Pmin
    interpolation = Smooth_Flux_Curves{i}(Idx_Amin{i}) + ...
    ((EXP_FILES_cell{i}(Idx_Amin{i}:Idx_Pmin{i}, 2) - EXP_FILES_cell{i}(Idx_Amin{i}, 2)) / (EXP_FILES_cell{i}(Idx_Pmin{i}, 2) -
    EXP_FILES_cell{i}(Idx_Amin{i}, 2))) * (Smooth_Flux_Curves{i}(Idx_Pmin{i}) - (Smooth_Flux_Curves{i}(Idx_Amin{i})));
    ## Extrapolación desde el punto de onset al punto de offset
    ExtrapolatedLines1{i} = interp1(EXP_FILES_cell{i}(Idx_Amin{i}: Idx_Pmin{i}, 2), InterpolatedLines1{i},
    EXP_FILES_cell{i}(Idx_Pmin{i}: Idx_Pmax{i}, 2), 'linear', 'extrap');

    ## Interpolación desde el punto Amax al punto de offset, Pmax
    InterpolatedLines2{i} = Smooth_Flux_Curves{i}(Idx_Pmax{i}) +
    ((EXP_FILES_cell{i}(Idx_Pmax{i}:Idx_Amax{i}, 2) - EXP_FILES_cell{i}(Idx_Pmax{i}, 2)) / (EXP_FILES_cell{i}(Idx_Amax{i}, 2) -
    EXP_FILES_cell{i}(Idx_Pmax{i}, 2))) * (Smooth_Flux_Curves{i}(Idx_Amax{i}) - (Smooth_Flux_Curves{i}(Idx_Pmax{i})));
    ## Extrapolación desde el punto de offset al punto de onset
    ExtrapolatedLines2{i} = interp1(EXP_FILES_cell{i}(Idx_Pmax{i}: Idx_Amax{i}, 2),
    InterpolatedLines2{i}, EXP_FILES_cell{i}(Idx_Pmin{i}: Idx_Pmax{i}, 2), 'linear', 'extrap');
    return baseline
