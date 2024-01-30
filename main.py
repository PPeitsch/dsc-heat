# Dependencies import
from core import files_names
from core import read_multiple_files
from core import graph_selection
from core import generate_baseline
import os

# os.walk is a generator
# calling next will get the first result in the form of a 3-tuple (dirpath, dirnames, filenames)
dir_name = '.'
root, dirs, files = next(os.walk(dir_name))
if '.ipynb_checkpoints' in dirs:
    dirs.remove('.ipynb_checkpoints')

directory_name = 'experimental'
# getting file names from directories
exp_list_names, experimental_files = files_names(dirs, directory_name)
# printing the list names to process
print(f'List of experimental files:\n{experimental_files}')
print(f'List of experimental files:\n{exp_list_names}')

directory_name = 'blank'
# getting file names from directories
blk_list_names, blank_files = files_names(dirs, directory_name)
# printing the list names to process
print(f'List of blank files:\n{blank_files}')
print(f'List of blank files:\n{blk_list_names}')

good_files_exp, bad_files_exp = read_multiple_files(experimental_files)
good_files_blank, bad_files_blank = read_multiple_files(blank_files)

# printing first 5 results
print(f"{good_files_exp[0].head(5)}\n")
print(f"{good_files_blank[0].head(5)}\n")

# heat flow vs time
x_label = 't(s)'
y_label = 'ΦQ(mW)'
directory_name = 'graficos'
figure_size = (12, 7.68)
to_graph = ('f', 't')
graph_selection(good_files_exp,
                directory_name,
                exp_list_names,
                figure_size,
                to_graph, x_label, y_label)

# heat flow vs temperature
x_label = 'T(°C)'
y_label = 'ΦQ(mW)'
directory_name = 'graficos'
figure_size = (12, 7.68)
to_graph = ('f', 'Tm')
graph_selection(good_files_exp,
                directory_name,
                exp_list_names,
                figure_size,
                to_graph, x_label, y_label)

# blank subtraction
files_sub = []
for n in range(len(good_files_exp)):
    files_sub.append(good_files_exp[n] - good_files_blank[n])
    files_sub[n].t = good_files_exp[n].t
    files_sub[n].Th = good_files_exp[n].Th
    files_sub[n].Tm = good_files_exp[n].Tm

# heat flow substracted vs temperature
x_label = 'T(°C)'
y_label = 'ΦQ(mW) sub'
directory_name = 'graficos'
figure_size = (12, 7.68)
to_graph = ('f', 'Tm')
graph_selection(files_sub,
                directory_name,
                exp_list_names,
                figure_size,
                to_graph, x_label, y_label)

# heat flow substracted vs time
x_label = 't(s)'
y_label = 'ΦQ(mW) sub'
directory_name = 'graficos'
figure_size = (12, 7.68)
to_graph = ('f', 't')
graph_selection(files_sub,
                directory_name,
                exp_list_names,
                figure_size,
                to_graph, x_label, y_label)

# heat flow substracted vs time, cut selection of the plot
x_label = 't(s)'
y_label = 'ΦQ(mW) sub - cutted'
directory_name = 'graficos'
figure_size = (12, 7.68)
to_graph = ('f', 't')
data_after_cut = graph_selection(files_sub,
                                 directory_name,
                                 exp_list_names,
                                 figure_size,
                                 to_graph, x_label, y_label,
                                 plot_cut=True)


baseline_curves = generate_baseline(data_after_cut)

# heat flow substracted vs time, cut selection of the plot including generated baseline
x_label = 't(s)'
y_label = 'ΦQ(mW) sub - cutted - bas'
directory_name = 'graficos'
figure_size = (12, 7.68)
to_graph = ('f', 't')
graph_selection(data_after_cut,
                directory_name,
                exp_list_names,
                figure_size,
                to_graph, x_label, y_label,
                plot_baseline=True,
                baseline_data=baseline_curves,
                show_plot=True)
