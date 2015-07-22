import argparse
import string
from copy import copy
from cwsm.spearmint import ConfigFile
import subprocess
import os

# command line arguments
parser = argparse.ArgumentParser(description='Run hyperparameter seacrh for a caffe model.')
parser.add_argument('--experiment', type=str, required=True, help='Exeriment root directory')
args = parser.parse_args()
netfile = args.experiment + '/model/trainval.prototxt'
solverfile = args.experiment + '/model/solver.prototxt'

# run checks that all the experiment folder structure is in place
def ensurepath(path):
    if not os.path.exists(path):
        os.mkdir(path)
        print 'Creating %s' % path

def demandpath(path, message):
    if not os.path.exists(path):
        print message % path
        exit()

ensurepath(args.experiment + '/caffeout')
ensurepath(args.experiment + '/data')
ensurepath(args.experiment + '/model')
ensurepath(args.experiment + '/mongodb')
ensurepath(args.experiment + '/spearmint')
ensurepath(args.experiment + '/tmp')

demandpath(args.experiment + '/data/mean_train.binaryproto', 'Please put mean files as %s')
demandpath(args.experiment + '/data/train_lmdb', 'Please put your training LMDB as %s')
demandpath(args.experiment + '/data/val_lmdb', 'Please put your validation LMDB as %s')
demandpath(args.experiment + '/model/solver.prototxt', 'Your Caffe solver file should be located at %s')
demandpath(args.experiment + '/model/trainval.prototxt', 'Your Caffe network description file should be located at %s')

# read in caffe .prototxt files
net = open(netfile, 'r').read()
solver = open(solverfile, 'r').read()

# check that solver has required placeholders
if '"PLACEHOLDER_NET"' not in solver:
    print 'Your solver.prototxt has to have "net: "PLACEHOLDER_NET"" line in it.'
    exit()
if '"PLACEHOLDER_MODEL_STORE"' not in solver:
    print 'Your solver.prototxt has to have "snapshot_prefix: "PLACEHOLDER_MODEL_STORE"" line in it.'
    exit()


# inialize templates for output files
tmpl_net = copy(net)
tmpl_solver = copy(solver)
smconfig = ConfigFile()

# parse OPTIMIZE tokens in the prototxt files into spearmint config.json
smconfig = ConfigFile()
smconfig.parse_in(net)
smconfig.parse_in(solver)
smconfig.footer()
smconfig.save(args.experiment)

# generate .prototxt templates
for i in range(1, len(smconfig.tokens) + 1):

    # replace OPTIMIZE{...} with OPTIMIZE_name in the .prototxt template file
    tmpl_net = string.replace(tmpl_net, smconfig.tokens[i]['description'], '_' + smconfig.tokens[i]['name'])
    tmpl_solver = string.replace(tmpl_solver, smconfig.tokens[i]['description'], '_' + smconfig.tokens[i]['name'], 1)

# store template files
with open(args.experiment + '/tmp/template_trainval.prototxt', 'w') as f:
    f.write(tmpl_net)
with open(args.experiment + '/tmp/template_solver.prototxt', 'w') as f:
    f.write(tmpl_solver)

# start Spearmint
subprocess.call("python ~/Software/Spearmint/spearmint/main.py %s/spearmint" % args.experiment, shell=True)

