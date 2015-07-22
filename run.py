import argparse
import string
from copy import copy
from cwsm.spearmint import ConfigFile
import subprocess

# command line arguments
parser = argparse.ArgumentParser(description='Run hyperparameter seacrh for a caffe model.')
parser.add_argument('--experiment', type=str, required=True, help='Exeriment root directory')
args = parser.parse_args()
netfile = args.experiment + '/model/trainval.prototxt'
solverfile = args.experiment + '/model/solver.prototxt'

# read in caffe .prototxt files
net = open(netfile, 'r').read()
solver = open(solverfile, 'r').read()

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

