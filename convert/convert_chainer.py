import os
import sys
import numpy as np
from net import *
from chainer import serializers
import itertools
import string


bn_map = {
    'c1':'b1',
    'c2':'b2',
    'c3':'b3',
    'd1':'b4',
    'd2':'b5',
    'r1_c1':'r1_b1',
    'r1_c2':'r1_b2',
    'r2_c1':'r2_b1',
    'r2_c2':'r2_b2',
    'r3_c1':'r3_b1',
    'r3_c2':'r3_b2',
    'r4_c1':'r4_b1',
    'r4_c2':'r4_b2',
    'r5_c1':'r5_b1',
    'r5_c2':'r5_b2',
}

class ChainerModelTree(object):
    def __init__(self, layers, parent_name = "", bn_map = {}):
        self.layers = [ChainerModelLayer(c, parent_name) for c in layers]
        self.bn_map = bn_map

    def empty(self):
        return len(self.layers) == 0

    def merge_bn(self):
        for layer in self.layers:
            # find the right BN layer, if any
            if layer.is_batch_normalization_layer():
                continue
            elif layer.is_residual_block():
                layer.children.merge_bn()
            else:
                bn_name = bn_map.get(layer.name,'')
                for other_layer in self.layers:
                    if bn_name == other_layer.name and other_layer.is_batch_normalization_layer():
                        layer.merge_bn(other_layer)

        # remove bn layers
        self.layers[:] = [ item for item in self.layers if not item.is_batch_normalization_layer() ]

    def dump(self, dst_path):
        s = ""
        for layer in self.layers:
            if layer.is_residual_block():
                s += layer.children.dump(dst_path)
            else:
                s += layer.dump(dst_path)
        return s


class ChainerModelLayer(object):
    def __init__(self, model_layer, parent_name = "", bn_map={}):
        self.model_layer = model_layer
        self.parent_name = parent_name
        if self.parent_name != "":
            self.parent_name += "_"
        self.name = self.parent_name + self.model_layer.name.replace("/", "_")
        self.children = ChainerModelTree(self.model_layer.children(), parent_name=self.name, bn_map=bn_map)
        self.post_bn_weights = None
        self.post_bn_bias = None
        # l = [[p[0].replace("/", "_"),p[1]] for p in children[0].model_layer.namedparams()]
        # a = [item for sublist in l for item in sublist]
        # self.params = dict(zip(a[0::2], a[1::2]))

    def is_batch_normalization_layer(self):
        return self.model_layer.__class__ == chainer.links.normalization.batch_normalization.BatchNormalization

    def is_residual_block(self):
        return self.model_layer.__class__ == ResidualBlock

    def is_convolutional_layer(self):
        return self.model_layer.__class__ == chainer.links.connection.convolution_2d.Convolution2D

    def is_deconvolutional_layer(self):
        return self.model_layer.__class__ == chainer.links.connection.deconvolution_2d.Deconvolution2D
    
    def has_children(self):
        return not self.children.empty

    def merge_bn(self, other):
        assert(other.is_batch_normalization_layer())
        if self.is_convolutional_layer():
            expander = (Ellipsis,) + (None,) * (self.model_layer.W.data.ndim - other.model_layer.gamma.ndim)
        else:
            expander = (None, Ellipsis) + (None,) * (self.model_layer.W.data.ndim - other.model_layer.gamma.ndim - 1)
        alpha = (other.model_layer.gamma.data / np.sqrt(other.var() + 0.001))
        print (self.name, other.name, self.model_layer.W.data.shape, alpha[expander].shape)
        self.post_bn_weights = self.model_layer.W.data * alpha[expander]
        self.post_bn_bias = other.model_layer.beta.data - (alpha * other.avg())

    def weights(self):
        assert(self.is_convolutional_layer() or self.is_deconvolutional_layer())
        if self.post_bn_weights is not None:
            return self.post_bn_weights
        else:
            return self.model_layer.W.data
        
    def bias(self):
        assert(self.is_convolutional_layer() or self.is_deconvolutional_layer())
        if self.post_bn_bias is not None:
            return self.post_bn_bias
        else:
            return self.model_layer.b.data

    def avg(self):
        assert(self.is_batch_normalization_layer())
        return self.model_layer.avg_mean

    def var(self):
        assert(self.is_batch_normalization_layer())
        return self.model_layer.avg_var

    def dump(self, dst_path):
        s = ""
        
        def convert(data):
            if data.ndim == 4:
                # Original VGG form (c_o, c_i, h, w) -> (c_o, h, w, c_i)
                data = data.transpose((0, 2, 3, 1))
            return data

        full_name = self.name + "_W"
        data = convert(self.weights())
        s += ("  modelParams[\"%s\"] = StyleModelData(modelName: modelName, rawFileName: \"%s\")\n" % (full_name, full_name))
        s += ("  //%s shape = %s\n" % (full_name, data.shape))
        g = open(dst_path + "/" + full_name + ".dat", "wb")
        data.ravel().tofile(g)
        g.close()

        full_name = self.name + "_b"
        data = convert(self.bias())
        s += ("  modelParams[\"%s\"] = StyleModelData(modelName: modelName, rawFileName: \"%s\")\n" % (full_name, full_name))
        s += ("  //%s shape = %s\n" % (full_name, data.shape))
        g = open(dst_path + "/" + full_name + ".dat", "wb")
        data.ravel().tofile(g)
        g.close()

        return s


class ChainerDataReader(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.model_name = os.path.splitext(os.path.basename(data_path))[0]
        self.model = FastStyleNet()
        self.load_using_chainer()
        self.cmt = ChainerModelTree(self.model.children())
        self.parameters = {}
        
    def load_using_chainer(self):
        print("Loading the chainer model.")
        serializers.load_npz(self.data_path, self.model)
        print("Done reading")
        
    def dump(self, dst_path):
        params = []

        self.cmt.merge_bn()
    
        s = self.cmt.dump(dst_path)
        
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
