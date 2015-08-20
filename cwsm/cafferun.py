import numpy as np
import cPickle
import math
import string
import re
import subprocess
from datetime import datetime
from cwsm.performance import Performance

def cafferun(params):

    # load general and optimization parameters
    with open('../tmp/optparams.pkl', 'rb') as f:
        paramdescr = cPickle.load(f)
    with open('../tmp/genparams.pkl', 'rb') as f:
        genparams = cPickle.load(f)
    CAFFE_ROOT = genparams['CAFFE_ROOT']
    optimize = genparams['optimize']
    optimizewrt = genparams['optimizewrt']

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
    trainnet = open('../tmp/template_trainval.prototxt', 'r').read()
    solver = open('../tmp/template_solver.prototxt', 'r').read()
    for p in params:
        trainnet = string.replace(trainnet, 'OPTIMIZE_' + p, str(params[p][0]), 1)
        solver = string.replace(solver, 'OPTIMIZE_' + p, str(params[p][0]), 1)

    # kappa optimizer has a special treatment
    if optimize == 'kappa':
        valnet = open('../tmp/template_val.prototxt', 'r').read()
        for p in params:
            valnet = string.replace(valnet, 'OPTIMIZE_' + p, str(params[p][0]), 1)

    # update paths for this run
    solver = string.replace(solver, 'PLACEHOLDER_NET', '../tmp/%s_trainval.prototxt' % prefix, 1)
    solver = string.replace(solver, 'PLACEHOLDER_MODEL_STORE', '../caffeout/%s' % prefix, 1)
    
    # store .prototxt for this run
    with open('../tmp/%s_trainval.prototxt' % prefix, 'w') as f:
        f.write(trainnet)
    if optimize == 'kappa':
        with open('../tmp/%s_val.prototxt' % prefix, 'w') as f:
            f.write(valnet)
    with open('../tmp/%s_solver.prototxt' % prefix, 'w') as f:
        f.write(solver)

    # run caffe training procedure
    caffe_return_code = subprocess.call(CAFFE_ROOT + '/build/tools/caffe train --solver ../tmp/%s_solver.prototxt 2> ../caffeout/%s_log.txt' % (prefix, prefix), shell=True)
    print 'CAFFE RETURN CODE ' + str(caffe_return_code)

    # set result to None by default
    result = None

    # if Caffe ran successfully update the result
    if int(caffe_return_code) == 0:

        # run the performace measure estimator
        if optimize == 'loss':
            result = Performance.loss(prefix, optimizewrt)
        elif optimize == 'accuracy':
            result = Performance.accuracy(prefix, optimizewrt)
        elif optimize == 'kappa':
            result = Performance.kappasq(prefix, CAFFE_ROOT, optimizewrt)
        else:
            print 'ERROR: Unknown perfomance measure %s' % optimize

    print '-----------------------------'
    print prefix, result
    print '-----------------------------'

    return result

# Write a function like this called 'main'
def main(job_id, params):
	return cafferun(params)

