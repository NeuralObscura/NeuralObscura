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
        useTemporary: Bool = false) {
        
        t1 = TensorDotLayer(
            kernelSize: kernelSize,
            channelsIn: channelsIn,
            channelsOut: channelsOut,
            w: w)
        
    }
    
    func chain(_ input: AnyCommandEncoder<MPSImage>) -> AnyCommandEncoder<MPSImage> {
        t1.chain(input)
        return AnyCommandEncoder<MPSImage>(self)
    }
    
    func forward(commandBuffer: MTLCommandBuffer) -> MPSImage {
        // TODO: This "as!" is to force compilation for testing purposes.
        return t1.forward(commandBuffer: commandBuffer) as! MPSImage
    }
    
    // private func destinationImageDescriptor(sourceImage: MPSImage) -> MPSImageDescriptor {
    //     let inHeight = sourceImage.height
    //     let inWidth = sourceImage.width

    //     let stride = self.stride
    //     
    //     let outHeight = stride * (inHeight - 1) + kernelSize - 2 * padding
    //     let outWidth = stride * (inWidth - 1) + kernelSize - 2 * padding

    //     /* Assert the constraint on input size, kernel size, padding, stride. */
    //     assert((outHeight + 2 * padding - kernelSize) % stride == 0,
    //            "Input size must be a multiple of i+2p-k in all dimensions. This constraint is failing in the height dimension.")
    //     assert((outWidth + 2 * padding - kernelSize) % stride == 0,
    //            "Input size must be a multiple of i+2p-k in all dimensions. This constraint is failing in the width dimension.")
    //     
    //     let descriptor = MPSImageDescriptor(
    //         channelFormat: textureFormat,
    //         width: Int(outWidth),
    //         height: Int(outHeight),
    //         featureChannels: channelsOut)
    //     
    //     return descriptor
    // }
}



