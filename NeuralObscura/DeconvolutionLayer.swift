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
class DeconvolutionLayer: UnaryCommandEncoder {
    init(
        kernelSize: Int,
        channelsIn: Int,
        channelsOut: Int,
        w: ParameterBuffer,
        b: ParameterBuffer,
        relu: Bool = true,
        padding: Int = 0, // TODO: Revisit this default
        stride: Int = 1,
        groupNum: Int = 1,
        debug: Bool = false) {
        super.init(
            delegate: DeconvolutionLayerDelegate(
                kernelSize: kernelSize,
                channelsIn: channelsIn,
                channelsOut: channelsOut,
                w: w,
                b: b,
                relu: relu,
                padding: padding,
                stride: stride),
            debug: debug)
    }
}

class DeconvolutionLayerDelegate: CommandEncoderDelegate {
    private let kernelSize: UInt32
    private let channelsIn: UInt32
    private let channelsOut: UInt32
    private let w: ParameterBuffer
    private let b: ParameterBuffer
    private let relu: Bool
    private let padding: Int
    private let stride: Int
    private var sourceImage: MPSImage!
    
    init(
        kernelSize: Int,
        channelsIn: Int,
        channelsOut: Int,
        w: ParameterBuffer,
        b: ParameterBuffer,
        relu: Bool,
        padding: Int,
        stride: Int) {
        
        self.kernelSize = UInt32(kernelSize)
        self.channelsIn = UInt32(channelsIn)
        self.channelsOut = UInt32(channelsOut)
        self.w = w
        self.b = b
        self.relu = relu
        self.padding = padding
        self.stride = stride
    }
    
    func getDestinationImageDescriptor(sourceImage: MPSImage) -> MPSImageDescriptor {
        let inHeight = sourceImage.height
        let inWidth = sourceImage.width

        let stride = self.stride
        
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
    
    
    func supplyInput(sourceImage: MPSImage, sourcePosition: Int) -> Bool {
        self.sourceImage = sourceImage
        return true
    }
    
    func encode(commandBuffer: MTLCommandBuffer, destinationImage: MPSImage) {
        let matrixWidth = sourceImage!.height * sourceImage!.width
        let matrixHeight = Int(channelsOut * kernelSize * kernelSize)
        var weightsShape = [channelsOut, kernelSize, kernelSize, channelsIn] as [UInt32]
        
        let weightsShapeBuffer = ShaderRegistry.getDevice().makeBuffer(
            bytes: &weightsShape,
            length: MemoryLayout<UInt32>.size * weightsShape.count,
            options: MTLResourceOptions.cpuCacheModeWriteCombined)
        let weightsBuffer = ShaderRegistry.getDevice().makeBuffer(
            bytes: self.w.pointer(),
            length: self.w.lengthInBytes(),
            options: MTLResourceOptions.cpuCacheModeWriteCombined)
        let tensordot = ShaderRegistry.getDevice().makeBuffer(
            length: MemoryLayout<Float32>.size * matrixWidth * matrixHeight,
            options: MTLResourceOptions.storageModePrivate)
        
        let tensordotPipelineState = ShaderRegistry.getOrLoad(name: "deconvolution_v2_tensordot")
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.setComputePipelineState(tensordotPipelineState)
        encoder.setTexture(sourceImage.texture, at: 0)
        encoder.setBuffer(tensordot, offset: 0, at: 1)
        encoder.setBuffer(weightsBuffer, offset: 0, at: 2)
        encoder.setBuffer(weightsShapeBuffer, offset: 0, at: 3)
        
        let threadGroupWidth = tensordotPipelineState.threadExecutionWidth
        let threadGroupHeight = tensordotPipelineState.maxTotalThreadsPerThreadgroup / threadGroupWidth
        let threadGroupShape = MTLSizeMake(threadGroupWidth, threadGroupHeight, 1)
        let gridShape = MTLSize(width: (matrixWidth + threadGroupWidth - 1) / threadGroupWidth,
                                          height: (matrixHeight + threadGroupHeight - 1) / threadGroupHeight,
                                          depth: 1)
        encoder.dispatchThreadgroups(gridShape, threadsPerThreadgroup: threadGroupShape)
        encoder.endEncoding()
        if let image = sourceImage as? MPSTemporaryImage {
            image.readCount -= 1
        }
    }
}



