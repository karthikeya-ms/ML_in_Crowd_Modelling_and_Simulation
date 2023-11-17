import sys
import os
src_path = '/'.join(os.getcwd().split('/')[:-1]) + '/src'
if src_path not in sys.path:
    sys.path.append('/'.join(sys.path[0].split('/')[:-1]) + '/src')