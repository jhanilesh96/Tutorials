import subprocess
import shutil
import sys, os
sys.path.insert(0, '..')
import utils as myutils


parseArg = myutils.parseArg
REMOVE = parseArg('rem', ofType=int, defValue=0, val_exprr=lambda t: (t>=0));
# remove only main training, keep pre-train
if REMOVE==1:
    try:        
        shutil.rmtree('./results_big/training_checkpoints/train')
    except:
        pass;

# remove all training
if REMOVE==2:
    try:        
        shutil.rmtree('./results_big/training_checkpoints')
    except:
        pass;

# remove all logs and training
if REMOVE==3:
    try:        
        shutil.rmtree('./results_big')
    except:
        pass;

for i in range(5):
    subprocess.run('python e01_v00_vrnn.py')


