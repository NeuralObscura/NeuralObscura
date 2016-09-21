import os
import sys
import numpy as np
from net import *
from chainer import serializers
import itertools

class ChainerDataReader(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.model_name = os.path.splitext(os.path.basename(data_path))[0]
        self.model = FastStyleNet()
        self.load_using_chainer()

    def load_using_chainer(self):
        print("Loading the chainer model.")
        serializers.load_npz(self.data_path, self.model)
        print("Done reading")
        children = [c for c in self.model.children()]

        def rename_layer(child_name, layer_name):
            return (self.model_name + '_' + child_name + layer_name).replace("/", "_")

        self.parameters = list(itertools.chain(*[[( rename_layer(child.name, param[0]), param[1].data) for param in child.namedparams()] for child in children]))


    def dump(self, dst_path):
        params = []
        def convert(data):
            if data.ndim == 4:
                # Original VGG form (c_o, c_i, h, w) -> (c_o, h, w, c_i)
                data = data.transpose((0, 2, 3, 1))
            return data

        s = ""
        for key, data in self.parameters:
            print(key)
            s += ("  let %s_Path = StyleModelData(modelName: \"%s\",rawFileName: \"%s\")\n" % (key, self.model_name, key))
            s += ("  //%s shape = %s\n" % (key, data.shape))

            # Save the individual files.
            g = open(dst_path + "/" + key + ".dat", "wb")
            convert(data).tofile(g)
            g.close()

        print("\nCopy this code:")
        print(s)
        print("Done!")

def main():
    args = sys.argv[1:]
    if len(args) != 2:
        print("usage: %s chainer.model output-folder" % os.path.basename(__file__))
        exit(-1)
    data_path, dst_path = args
    ChainerDataReader(data_path).dump(dst_path)

if __name__ == '__main__':
    main()
