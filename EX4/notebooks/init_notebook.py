"""
This is a helper module. If a notebook imports this module
will have access to direct imports from all modules in
the src directory. In adition, the working directory is 
changed to the data directory. This means that all loading
and saving of files is done in relation to the data directory.
"""
import sys
import os

exercise_directory = os.path.dirname(os.getcwd())

src_path = exercise_directory + '/src'

if src_path not in sys.path:
    sys.path.append(src_path)

# Do this if you want to change the working directory to data
os.chdir(exercise_directory + '/data')
