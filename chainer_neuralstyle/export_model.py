#!/usr/bin/env python

import os
import sys
import numpy as np
from net import *
from chainer import serializers
import itertools

def flatten(iter_of_iters):
    return itertools.chain(*iter_of_iters)

class ChainerDataReader(object):
    def __init__(self, data_path):
        self.model_name = os.path.splitext(os.path.basename(data_path))[0]

        model = FastStyleNet()
        serializers.load_npz(data_path, model)

        def extract(link, param):
            name = (link.name + param[0]).replace("/", "_")
            data = param[1].data
            if data.ndim == 4:
                if isinstance(link, L.Deconvolution2D):
                    # Deconv has a weight shape of (c_i, c_o, k_h, k_w) in chainer
                    # MPS CNN Conv expects (c_o, k_h, k_w, c_i)
                    data = data.transpose(1, 2, 3, 0)
                else:
                    # Conv has a weight shape of (c_o, c_i, k_h, k_w) in chainer
                    # MPS CNN Conv expects (c_o, k_h, k_w, c_i)
                    data = data.transpose(0, 2, 3, 1)
            return name, data
        self.params = flatten([[extract(link, param) for param in link.namedparams()] for link in model.children()])


    def dump(self, dst_path):
        s = ""
        for name, data in self.params:
            s += ("  modelParams[\"%s\"] = StyleModelData(modelName: modelName, rawFileName: \"%s\")\n" % (name, name))
            s += ("  //%s shape = %s\n" % (name, data.shape))
            # Save the individual files.
            g = open(dst_path + "/" + name + ".dat", "wb")
            data.ravel().tofile(g)
            g.close()
        print("\nCopy this code:")
        print(s)

def main():
    args = sys.argv[1:]
    if len(args) != 2:
        print("usage: %s chainer.model output-folder" % os.path.basename(__file__))
        exit(-1)
    data_path, dst_path = args
    ChainerDataReader(data_path).dump(dst_path)

if __name__ == '__main__':
    main()
