import argparse
import string
from copy import copy
from cwsm.spearmint import ConfigFile
import subprocess
import os
import cPickle
import time

# binary locations
CAFFE_ROOT = '/home/hpc_kuz/Software/Caffe'  # without the trailing slash
SPEARMINT_ROOT = '/home/hpc_kuz/Software/Spearmint'  # without the trailing slash
MONGODB_BIN = '/home/hpc_kuz/Software/mongodb/mongodb-linux-x86_64-3.0.4/bin/mongod'

# command line arguments
parser = argparse.ArgumentParser(description='Run hyperparameter seacrh for a caffe model.')
parser.add_argument('--experiment', type=str, required=True, help='Exeriment root directory')
parser.add_argument('--optimize', type=str, required=True, help='Performance measure to optimize: loss, accuracy, kappa')
parser.add_argument('--optimizewrt', type=str, required=True, help='Performance results can be reported from "last" or from the "best" iteration within each run')
args = parser.parse_args()
trainnetfile = args.experiment + '/model/trainval.prototxt'
valnetfile = args.experiment + '/model/val.prototxt'
solverfile = args.experiment + '/model/solver.prototxt'

# check that PYTHONPATH has CWSM root
if os.getcwd() not in os.environ['PYTHONPATH'].split(':'):
    print 'CWSM root directory is not in $PYTHONPATH, please run\nexport PYTHONPATH=%s:$PYTHONPATH' % os.getcwd()
    exit()

# run checks that all the experiment folder structure is in place and third-party software is available
def ensurepath(path):
    if not os.path.exists(path):
        os.mkdir(path)
        print 'Creating %s' % path

def demandpath(path, message):
    if not os.path.exists(path):
        print message % path
        exit()

def warningpath(path, message):
    if not os.path.exists(path):
        print message % path

ensurepath(args.experiment + '/caffeout')
ensurepath(args.experiment + '/data')
ensurepath(args.experiment + '/model')
ensurepath(args.experiment + '/mongodb')
ensurepath(args.experiment + '/spearmint')
ensurepath(args.experiment + '/tmp')

demandpath(CAFFE_ROOT + '/build/tools/caffe', 'Caffe binary not found at %s. Set the CAFFE_ROOT variable.')
demandpath(SPEARMINT_ROOT + '/spearmint/main.py', 'Spearmint main script was not found at %s. Set the SPEARMINT_ROOT variable.')
demandpath(SPEARMINT_ROOT + '/spearmint/cleanup.sh', 'Spearmint cleanup script was not found at %s. Set the SPEARMINT_ROOT variable.')
demandpath(MONGODB_BIN, 'MongoDB is not installed? Server binary not found at %s.')

warningpath(args.experiment + '/data/mean_train.binaryproto', 'WARNING: image mean file was not found: %s. Ignore this message if you know that you do not need it.')
demandpath(args.experiment + '/data/train_lmdb', 'Please put your training LMDB as %s.')
demandpath(args.experiment + '/data/val_lmdb', 'Please put your validation LMDB as %s.')
demandpath(args.experiment + '/model/solver.prototxt', 'Your Caffe solver file should be located at %s.')
demandpath(args.experiment + '/model/trainval.prototxt', 'Your Caffe network description file should be located at %s.')

# run MongoDB
subprocess.call('pkill mongod', shell=True)
time.sleep(2)
subprocess.call('%s --fork --logpath %s/mongodb/log.txt --dbpath %s/mongodb' % (MONGODB_BIN, args.experiment, args.experiment), shell=True)

# clearn previous results
print 'Removing previous results and temporary files ...'
subprocess.call('bash ' + SPEARMINT_ROOT + '/spearmint/cleanup.sh' + ' ' + args.experiment + '/spearmint', shell=True)
subprocess.call('rm ' + args.experiment + '/caffeout/*', shell=True)
subprocess.call('rm -r ' + args.experiment + '/spearmint/*', shell=True)
subprocess.call('rm -r ' + args.experiment + '/tmp/*', shell=True)

# store genral parameters for the future use
genparams = {}
genparams['CAFFE_ROOT'] = CAFFE_ROOT
genparams['SPEARMINT_ROOT'] = SPEARMINT_ROOT
genparams['optimize'] = args.optimize
genparams['optimizewrt'] = args.optimizewrt

# read in caffe .prototxt files
trainnet = open(trainnetfile, 'r').read()
solver = open(solverfile, 'r').read()

# check that solver has required placeholders
if '"PLACEHOLDER_NET"' not in solver:
    print 'Your solver.prototxt has to have "net: "PLACEHOLDER_NET"" line in it.'
    exit()
if '"PLACEHOLDER_MODEL_STORE"' not in solver:
    print 'Your solver.prototxt has to have "snapshot_prefix: "PLACEHOLDER_MODEL_STORE"" line in it.'
    exit()

# check that trainval has the accuracy or loss layer
if args.optimize == 'accuracy' and 'name: "accuracy"' not in trainnet:
    print 'Your trainval.prototxt has to have a layer with name: "accuracy".'
    exit()
if args.optimize == 'loss' and 'name: "loss"' not in trainnet:
    print 'Your trainval.prototxt has to have a layer with name: "loss".'
    exit()

# if we want to optimize kappa there are speical requirements
if args.optimize == 'kappa':
    demandpath(args.experiment + '/model/val.prototxt', 'Your Caffe network validation description file should be located at %s.')
    demandpath(args.experiment + '/data/val_labels.txt', 'To use kappa measure you need to have true labels stored as %s. Each line of this file should be in the form "filename.jpg 2".')
    valnet = open(valnetfile, 'r').read()
    if 'name: "prob"' not in valnet:
        print 'Your val.prototxt has to have layer with name: "prob".'
    tmpl_valnet = copy(valnet)

# inialize templates for output files
tmpl_trainnet = copy(trainnet)
tmpl_solver = copy(solver)
smconfig = ConfigFile()

# parse OPTIMIZE tokens in the prototxt files into spearmint config.json
smconfig = ConfigFile()
smconfig.parse_in(trainnet)
smconfig.parse_in(solver)
smconfig.footer()
smconfig.save(args.experiment)

# move caffe function optimizer to the Spermint experimnet directory
subprocess.call('cp cwsm/cafferun.py %s/spearmint' % args.experiment, shell=True)

# store general parameters
with open(args.experiment + '/tmp/genparams.pkl', 'wb') as f:
    cPickle.dump(genparams, f)

# generate .prototxt templates
for i in range(1, len(smconfig.tokens) + 1):

    # replace OPTIMIZE{...} with OPTIMIZE_name in the .prototxt template file
    tmpl_trainnet = string.replace(tmpl_trainnet, smconfig.tokens[i]['description'], '_' + smconfig.tokens[i]['name'], 1)
    tmpl_solver = string.replace(tmpl_solver, smconfig.tokens[i]['description'], '_' + smconfig.tokens[i]['name'], 1)
    if args.optimize == 'kappa':
        tmpl_valnet = string.replace(tmpl_valnet, smconfig.tokens[i]['description'], '_' + smconfig.tokens[i]['name'], 1)

# store template files
with open(args.experiment + '/tmp/template_trainval.prototxt', 'w') as f:
    f.write(tmpl_trainnet)
with open(args.experiment + '/tmp/template_solver.prototxt', 'w') as f:
    f.write(tmpl_solver)
if args.optimize == 'kappa':
    with open(args.experiment + '/tmp/template_val.prototxt', 'w') as f:
        f.write(tmpl_valnet)

# start Spearmint
subprocess.call("python %s/spearmint/main.py %s/spearmint" % (SPEARMINT_ROOT, args.experiment), shell=True)

