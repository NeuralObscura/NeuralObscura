//
//  DeconvolutionLayer.swift
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
class DeconvolutionLayer: CommandEncoder {
    init(
        channelsIn: Int,
        channelsOut: Int,
        kernelSize: Int,
        w: ParameterBuffer,
        b: ParameterBuffer,
        neuronFilter: MPSCNNNeuron? = nil,
        padding: Int = 0, // TODO: Revisit this default
        stride: Int = 1,
        destinationFeatureChannelOffset: Int = 0,
        groupNum: Int = 1,
        outputType: CommandEncoderOutputType = CommandEncoderOutputType.debug) {
        super.init(
            delegate: DeconvolutionLayerDelegate(
                channelsIn: channelsIn,
                channelsOut: channelsOut,
                kernelSize: kernelSize,
                w: w,
                b: b,
                neuronFilter: neuronFilter,
                padding: padding,
                stride: stride,
                destinationFeatureChannelOffset: destinationFeatureChannelOffset,
                groupNum: groupNum),
            outputType: outputType)
    }
}

class DeconvolutionLayerDelegate: CommandEncoderDelegate {
    private let convolution: MPSCNNConvolution
    private let padding: Int
    private let stride: Int
    private let interpixelStride: MTLBuffer?
    
    init(
        channelsIn: Int,
        channelsOut: Int,
        kernelSize: Int,
        w: ParameterBuffer,
        b: ParameterBuffer,
        neuronFilter: MPSCNNNeuron? = MPSCNNNeuronReLU(),
        padding: Int  = 0,
        stride: Int = 1,
        destinationFeatureChannelOffset: Int = 0,
        groupNum: Int = 1) {
        
        self.padding = padding
        self.stride = stride
        
        if stride > 1 {
            var s = UInt8(stride)
            self.interpixelStride = ShaderRegistry.getDevice().makeBuffer(
                bytes: &s,
                length: MemoryLayout<UInt8>.size,
                options: MTLResourceOptions.cpuCacheModeWriteCombined)
        } else { interpixelStride = nil }
        
        // create appropriate convolution descriptor with appropriate stride
        let convDesc = MPSCNNConvolutionDescriptor(kernelWidth: kernelSize,
                                                   kernelHeight: kernelSize,
                                                   inputFeatureChannels: channelsIn,
                                                   outputFeatureChannels: channelsOut,
                                                   neuronFilter: neuronFilter)
        
        /* The effective stride is 1, because we're performing deconvolution with a convolution layer */
        convDesc.strideInPixelsX = 1
        convDesc.strideInPixelsY = 1
        
        assert((groupNum > 0), "Group size can't be less than 1")
        convDesc.groups = Int(groupNum)
        
        // initialize the convolution layer by calling the parent's (MPSCNNConvlution's) initializer
        convolution = MPSCNNConvolution(
            device: ShaderRegistry.getDevice(),
            convolutionDescriptor: convDesc,
            kernelWeights: w.pointer(),
            biasTerms: b.pointer(),
            flags: MPSCNNConvolutionFlags.none)
        convolution.destinationFeatureChannelOffset = destinationFeatureChannelOffset
        convolution.edgeMode = .zero
        let effectivePadding = kernelSize - padding - 1
        convolution.offset = MPSOffset(x: 1 - effectivePadding, y: 1 - effectivePadding, z: 0)
    }
    
    func getDestinationImageDescriptor(sourceImage: MPSImage) -> MPSImageDescriptor {
        let inHeight = sourceImage.height
        let inWidth = sourceImage.width
        
        let kernelSize = convolution.kernelWidth
        let stride = self.stride
        let channelsOut = convolution.outputFeatureChannels
        
        let outHeight = stride * (inHeight - 1) + kernelSize - 2 * padding
        let outWidth = stride * (inWidth - 1) + kernelSize - 2 * padding
        
        /* Assert the constraint on input size, kernel size, padding, stride. */
        assert((outHeight + 2 * padding - kernelSize) % stride == 0,
               "Input size must be a multiple of i+2p-k in all dimensions. This constraint is failing in the height dimension.")
        assert((outWidth + 2 * padding - kernelSize) % stride == 0,
               "Input size must be a multiple of i+2p-k in all dimensions. This constraint is failing in the width dimension.")
        
        let descriptor = MPSImageDescriptor(
            channelFormat: textureFormat,
            width: outWidth,
            height: outHeight,
            featureChannels: channelsOut)
        
        return descriptor
    }
    
    func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage) {
        
        /* We don't need interpixel stride if the stride is 1 */
        if let _ = interpixelStride {
            let encoder = commandBuffer.makeComputeCommandEncoder()
            encoder.setComputePipelineState(ShaderRegistry.getOrLoad(name: "deconvolution_interpixel_stride"))
            encoder.setTexture(sourceImage.texture, at: 0)
            encoder.setTexture(destinationImage.texture, at: 1)
            encoder.setBuffer(interpixelStride!, offset: 0, at: 2)
            // TODO: Optimize
            let threadsPerGroups = MTLSizeMake(1, 1, 1)
            let threadGroups = MTLSizeMake(
                destinationImage.texture.width,
                destinationImage.texture.height,
                destinationImage.texture.arrayLength)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroups)
            encoder.endEncoding()
        }
        
        // encode standard convolution
        convolution.encode(commandBuffer: commandBuffer, sourceImage: sourceImage, destinationImage: destinationImage)
    }
}



