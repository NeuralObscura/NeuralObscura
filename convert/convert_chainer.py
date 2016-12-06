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
            return (child_name + layer_name).replace("/", "_")

        self.parameters = []
        for child in children:
            if child.__class__ == chainer.links.normalization.batch_normalization.BatchNormalization:
                self.parameters.append((rename_layer(child.name, '_mean'), child.avg_mean))
                self.parameters.append((rename_layer(child.name, '_stddev'), np.sqrt(child.avg_var)))
            if child.__class__ == ResidualBlock:
                for block_child in child.children():
                    if block_child.__class__ == chainer.links.normalization.batch_normalization.BatchNormalization:
                        self.parameters.append((rename_layer(child.name + '_' + block_child.name, '_mean'), block_child.avg_mean))
                        self.parameters.append((rename_layer(child.name + '_' + block_child.name, '_stddev'), np.sqrt(block_child.avg_var)))
            for param in child.namedparams():
                self.parameters.append((rename_layer(child.name, param[0]), param[1].data))


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
            data = convert(data)
            s += ("  modelParams[\"%s\"] = FileParameterBuffer(modelName: modelName, rawFileName: \"%s\")\n" % (key, key))
            s += ("  //%s shape = %s\n" % (key, data.shape))

            # Save the individual files.
            g = open(dst_path + "/" + key + ".dat", "wb")
            data.ravel().tofile(g)
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
