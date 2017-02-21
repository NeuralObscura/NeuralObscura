//
//  DeconvolutionBlock.swift
//  NeuralObscura
//
//  Created by Edward Knox on 9/25/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

/**
 Computes a deconvolution operation.
 
 Our reference implementation found at: https://arxiv.org/pdf/1603.07285.pdf
 
 Our implementation applies the constraint where in both dimensions the input size is a multiple
 of `i + 2p - k` where `i` is the input size in that dimension, `p` is the padding in that dimension,
 and `k` is the kernel size in that dimension.
 */
class DeconvolutionBlock: UnaryCommandEncoder {
    private let t1: TensorDotLayer
    
    init(
        kernelSize: Int,
        channelsIn: Int,
        channelsOut: Int,
        w: ParameterBuffer,
        b: ParameterBuffer,
        relu: Bool = true,
        padding: Int = 0, // TODO: Revisit this default
        stride: Int = 1,
        debug: Bool = false) {
        
        t1 = TensorDotLayer(
            kernelSize: kernelSize,
            channelsIn: channelsIn,
            channelsOut: channelsOut,
            w: w,
            debug: debug)
        
    }
    
    func chain(_ input: AnyCommandEncoder<MPSImage>) -> AnyCommandEncoder<MPSImage> {
        return t1.chain(input)
    }
    
    func forward(commandBuffer: MTLCommandBuffer) -> MPSImage {
    }
}



