"""

Model performance estimators for the Caffe

"""

import re
import subprocess
from cwsm.lmdbtools import LMDBTools
import numpy as np


class Performance:

    @staticmethod
    def loss(prefix):
        logbuffer = open('../caffeout/%s_log.txt' % prefix, 'r').read()
        pattern = re.compile('Test net output #[0-9]+: loss = [0-9]+\.[0-9]+')
        matches = re.findall(pattern, logbuffer)
        result = float(re.search('[0-9]+\.[0-9]+', matches[-1]).group(0))
        return result

    @staticmethod
    def accuracy(prefix):
        logbuffer = open('../caffeout/%s_log.txt' % prefix, 'r').read()
        pattern = re.compile('Test net output #[0-9]+: accuracy = [0-9]+\.[0-9]+')
        matches = re.findall(pattern, logbuffer)
        result = float(re.search('[0-9]+\.[0-9]+', matches[-1]).group(0))
        
        # Spearmint by default is trying to minimize, therefore return -accuracy
        return -result

    @staticmethod
    def kappasq(prefix, CAFFE_ROOT):
        
        # remove previous features
        subprocess.call('rm -r ../tmp/features', shell=True)

        # find the most recent .caffemodel for the given prefix
        lastiter = subprocess.check_output("ls -1 ../caffeout/%s*.caffemodel | awk -F '[_.]' '{print $5}' | sort -n | tail -n 1" % prefix, shell=True).strip()

        # number of samples in the validation set
        nval = int(subprocess.check_output("cat ../data/val_labels.txt | wc -l", shell=True).strip())

        # extract features
        subprocess.call('%s/build/tools/extract_features.bin ../caffeout/%s_iter_%s.caffemodel ../tmp/%s_val.prototxt prob ../tmp/features %d lmdb GPU 0' % (CAFFE_ROOT, prefix, lastiter, prefix, nval), shell=True)

        # convert features to predictions
        predicted_dict = LMDBTools.extract_predictions('../tmp/features')

        # read in actual validation labels
        actual = []
        predicted = []
        for (filename, cls) in [x.split() for x in open('../data/val_labels.txt', 'r').read().strip().split('\n')]:
            actual.append(int(cls))
            predicted.append(int(predicted_dict[filename]))

        #
        # calculate kappa
        #
        squared = True
        ratings = np.vstack((actual, predicted)).T
        categories = int(np.amax(ratings)) + 1
        subjects = ratings.size / 2

        # build weight matrix
        weighted = np.empty((categories, categories))
        for i in range(categories):
            for j in range(categories):
                weighted[i, j] = abs(i - j) ** 2

        # build observed matrix
        observed = np.zeros((categories, categories))
        distributions = np.zeros((categories, 2))
        for k in range(subjects):
            observed[ratings[k, 0], ratings[k, 1]] += 1
            distributions[ratings[k, 0], 0] += 1
            distributions[ratings[k, 1], 1] += 1
        
        # normalize observed and distribution arrays
        observed = observed / subjects
        distributions = distributions / subjects
        
        # build expected array
        expected = np.empty((categories, categories))
        for i in range(categories):
            for j in range(categories):
                expected[i, j] = distributions[i, 0] * distributions[j, 1]
        
        # calculate kappa
        kappa = 1.0 - (sum(sum(weighted * observed)) / sum(sum(weighted * expected)))

        # Spearmint wants to minimize, so return negative value of kappa (effectively requiring to maximize kappe instead)
        return -kappa

