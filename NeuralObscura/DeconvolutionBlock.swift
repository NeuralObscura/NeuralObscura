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
 
    private let tensordot: TensorDotLayer
    private let col2im: Col2ImLayer
    
    init(
        kernelSize: Int,
        channelsIn: Int,
        channelsOut: Int,
        w: ParameterBuffer,
        b: ParameterBuffer,
        relu: Bool = true,
        padding: Int = 0, // TODO: Revisit this default
        stride: Int = 1,
        useTemporary: Bool = false) {
        
        tensordot = TensorDotLayer(
            kernelSize: kernelSize,
            channelsIn: channelsIn,
            channelsOut: channelsOut,
            w: w)
        // TODO: inputSize, outputRowWidth params are set to 0 so we can compile until we know how to populate them
        col2im = Col2ImLayer(channelsOut: UInt32(channelsOut),
                             kernelSize: UInt32(kernelSize),
                             stride: UInt32(stride),
                             padding: UInt32(padding))
    }
    
    func chain(_ input: AnyCommandEncoder<MPSImage>) -> AnyCommandEncoder<MPSImage> {
        col2im.chain(tensordot.chain(input), input)
        return AnyCommandEncoder<MPSImage>(self)
    }
    
    func forward(commandBuffer: MTLCommandBuffer) -> MPSImage {
        return col2im.forward(commandBuffer: commandBuffer)
    }
    
    func registerConsumer() {
        col2im.registerConsumer()
    }
}
