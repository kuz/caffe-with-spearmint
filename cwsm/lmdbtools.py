"""

Tools for handling LMDB

"""

import lmdb
import caffe
import numpy as np

class LMDBTools:

    @staticmethod
    def extract_predictions(features_path):
        
        env = lmdb.open(features_path)
        env_stat = env.stat()
        num_images = env_stat['entries']

        prediction_dict = {}
        with env.begin() as txn:
            with txn.cursor() as curs:
                for key, value in curs:
                    datum = caffe.proto.caffe_pb2.Datum()
                    datum.ParseFromString(value)
                    arr = caffe.io.datum_to_array(datum)
                    prediction_dict[int(key)] = np.argmax(arr[:, 0, 0])

        predictions = {}
        with open('../data/val_labels.txt', 'r') as f:
            for i, line in enumerate(f):
                predictions[line.split(' ')[0]] = prediction_dict[i]

        return predictions

