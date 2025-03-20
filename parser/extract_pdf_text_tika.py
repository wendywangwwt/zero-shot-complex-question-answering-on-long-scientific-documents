import os
import argparse
import json
import tika
tika.initVM() # initialize the java server for tika

from tika import parser

# Instantiate the parser
parser_argparse = argparse.ArgumentParser(description='Optional app description')

# Required positional argument
parser_argparse.add_argument('--dir_data', type=str, default='.',
                    help='directory of zst files to filter')

args = parser_argparse.parse_args()
dir_data = args.dir_data

# get all filenames under data dirctory
def get_fns(dir_files, extension='zst'):
    """
    get qualified filenames under a given directory
    """
    return [f for f in os.listdir(dir_files) if os.path.isfile(os.path.join(dir_files, f)) and f.endswith('.'+extension)]

fns = get_fns(dir_data, 'pdf')
print(f'Found {len(fns)} pdf files')

count = 0
for fn in fns:
    path = os.path.join(dir_data,fn)
    res = parser.from_file(path, headers={'X-Tika-PDFenableAutoSpace': 'true'})
    
    with open(path.replace('.pdf','.json'),'w') as f:
        json.dump(res, f)
    
    count += 1
    
    if count % 10 == 0 or count == len(fns):
        print(f'Done {count}/{len(fns)}')

print('Finished!')