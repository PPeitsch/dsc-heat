# Dependencies import
from core import files_names
from core import read_multiple_files
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
blk_list_names, blank_files = files_names(dirs, directory_name)
print(f'List of blank files:\n{blank_files}')
print(f'List of blank files:\n{blk_list_names}')

good_files_exp, bad_files_exp = read_multiple_files(experimental_files)
good_files_blank, bad_files_blank = read_multiple_files(blank_files)

# printing first 5 results
print(f"{good_files_exp[0].head(5)}\n")
print(f"{good_files_blank[0].head(5)}\n")



