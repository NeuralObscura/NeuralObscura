//
//  TensorDotLayer.swift
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
class TensorDotLayer: UnaryCommandEncoder {
    var input: AnyCommandEncoder<MPSImage>!
    
    private let kernelSize: Int
    private let channelsIn: Int
    private let channelsOut: Int
    private let w: ParameterBuffer
    private var sourceImage: MPSImage!
    private var outputMemoId: Int?
    private var outputMemo: MTLBuffer?
    
    init(
        kernelSize: Int,
        channelsIn: Int,
        channelsOut: Int,
        w: ParameterBuffer) {
        self.kernelSize = kernelSize
        self.channelsIn = channelsIn
        self.channelsOut = channelsOut
        self.w = w
    }
    
    func chain(_ input: AnyCommandEncoder<MPSImage>) -> AnyCommandEncoder<MTLBuffer> {
        self.input = input
        input.registerConsumer()
        return AnyCommandEncoder<MTLBuffer>(self)
    }
    
    func registerConsumer() {}
    
    func forward(commandBuffer: MTLCommandBuffer) -> MTLBuffer {
        if outputMemoId != nil && outputMemoId! == commandBuffer.hash {
            return outputMemo!
        } else {
            let sourceImage = input.forward(commandBuffer: commandBuffer)
            let matrixWidth = sourceImage.height * sourceImage.width
            let matrixHeight = channelsOut * kernelSize * kernelSize
            var weightsShape = [
                UInt32(channelsOut),
                UInt32(kernelSize),
                UInt32(kernelSize),
                UInt32(channelsIn)]
            let weightsShapeBuffer = ShaderRegistry.getDevice().makeBuffer(
                bytes: &weightsShape,
                length: MemoryLayout<UInt32>.size * weightsShape.count,
                options: MTLResourceOptions.cpuCacheModeWriteCombined)
            let weightsConverted = Conversions.float32toFloat16(
                pointer: UnsafeMutableRawPointer(mutating: self.w.pointer),
                count: self.w.count)
            let weightsBuffer = ShaderRegistry.getDevice().makeBuffer(
                bytes: weightsConverted,
                length: weightsConverted.count * ExpectedFloat16Size,
                options: MTLResourceOptions.cpuCacheModeWriteCombined)
            let tensordot = ShaderRegistry.getDevice().makeBuffer(
                length: ExpectedFloat16Size * matrixWidth * matrixHeight,
                options: MTLResourceOptions.cpuCacheModeWriteCombined)
            let tensordotPipelineState = ShaderRegistry.getOrLoad(name: "tensordot")
            let encoder = commandBuffer.makeComputeCommandEncoder()
            encoder.setComputePipelineState(tensordotPipelineState)
            encoder.setTexture(sourceImage.texture, at: 0)
            encoder.setBuffer(tensordot, offset: 0, at: 1)
            encoder.setBuffer(weightsBuffer, offset: 0, at: 2)
            encoder.setBuffer(weightsShapeBuffer, offset: 0, at: 3)
            let threadGroupWidth = tensordotPipelineState.threadExecutionWidth
            let threadGroupHeight = tensordotPipelineState.maxTotalThreadsPerThreadgroup / threadGroupWidth
            let threadGroupShape = MTLSizeMake(threadGroupWidth, threadGroupHeight, 1)
            let gridShape = MTLSize(
                width: (matrixWidth + threadGroupWidth - 1) / threadGroupWidth,
                height: (matrixHeight + threadGroupHeight - 1) / threadGroupHeight,
                depth: 1)
            encoder.dispatchThreadgroups(gridShape, threadsPerThreadgroup: threadGroupShape)
            encoder.endEncoding()
            
            if let image = sourceImage as? MPSTemporaryImage {
                image.readCount -= 1
            }
            
            outputMemoId = commandBuffer.hash
            outputMemo = tensordot
            return outputMemo!
        }
    }
}



