import numpy as np
import cPickle
import math
import string
import re
import subprocess
from datetime import datetime

def cafferun(params):

    # load full parameter descriptions
    with open('../tmp/parameters.pkl', 'rb') as f:
        paramdescr = cPickle.load(f)

    # transform parameters accoring to transformation specified in the model file
    print params
    for p in params:
        if paramdescr[p].get('transform', None) is not None:

            # X<>: multiplier where <> stands for any number (examples: X10, X100, X22)
            if paramdescr[p]['transform'][0] == 'X':
                multiplier = int(paramdescr[p]['transform'][1:])
                params[p][0] *= multiplier

            # LOG<>: number which goes to Spearmint corresponds to log with base <> of an actual
            #        number (example: value 2 of LOG10 corresponds to 100)
            if paramdescr[p]['transform'][0:3] == 'LOG':
                base = int(paramdescr[p]['transform'][3:])
                params[p][0] = math.log(params[p][0], base)
            
            # NEGEXP<>: where <> is  the base, the number which goes to Spearmint is negative of the 
            #           exponent (example: value 3 with NEGEXP10 means 10^-3 and correpsonds to 0.001)
            if paramdescr[p]['transform'][0:6] == 'NEGEXP':
                negexp = float(paramdescr[p]['transform'][6:])
                params[p] = [negexp ** float(-params[p][0])]

    # unique prefix for this run
    prefix = datetime.now().strftime('%Y-%d-%m-%H-%M-%S')

	# generate .prototxt files with current set of paramters
    net = open('../tmp/template_trainval.prototxt', 'r').read()
    solver = open('../tmp/template_solver.prototxt', 'r').read()
    for p in params:
        net = string.replace(net, 'OPTIMIZE_' + p, str(params[p][0]), 1)
        solver = string.replace(solver, 'OPTIMIZE_' + p, str(params[p][0]), 1)

    # update paths for this run
    solver = string.replace(solver, 'PLACEHOLDER_NET', '../tmp/%s_trainval.prototxt' % prefix, 1)
    solver = string.replace(solver, 'PLACEHOLDER_MODEL_STORE', '../caffeout/%s' % prefix, 1)
    
    # store .prototxt for this run
    with open('../tmp/%s_trainval.prototxt' % prefix, 'w') as f:
        f.write(net)
    with open('../tmp/%s_solver.prototxt' % prefix, 'w') as f:
        f.write(solver)

    # run caffe training procedure
    caffe_return_code = subprocess.call("~/Software/Caffe/build/tools/caffe train --solver ../tmp/%s_solver.prototxt 2> ../caffeout/%s_log.txt" % (prefix, prefix), shell=True)
    print 'CAFFE RETURN CODE ' + str(caffe_return_code)

    # set result to None by default
    result = None

    # if Caffe ran successfully update the result
    if int(caffe_return_code) == 0:

        # run the performace measure estimator
        logbuffer = open('../caffeout/%s_log.txt' % prefix, 'r').read()
        pattern = re.compile('Test net output #1: loss = [0-9]+\.[0-9]+')
        matches = re.findall(pattern, logbuffer)
        result = float(re.search('[0-9]+\.[0-9]+', matches[-1]).group(0))

    print '-----------------------------'
    print prefix, result
    print '-----------------------------'

    return result

# Write a function like this called 'main'
def main(job_id, params):
	return cafferun(params)

