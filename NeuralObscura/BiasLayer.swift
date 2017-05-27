//
//  BiasLayer.swift
//  NeuralObscura
//
//  Created by Edward Knox on 9/27/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class BiasLayer: UnaryCommandEncoder {
    private let useTemporary: Bool
    private var consumerCount: Int = 0
    private var biases: ParameterBuffer
    private var input: AnyCommandEncoder<MPSImage>!
    private var outputMemoId: Int?
    private var outputMemo: MPSImage?
    
    init(biases: ParameterBuffer,
        useTemporary: Bool = false) {
        self.useTemporary = useTemporary
        self.biases = biases
    }
    
    func chain(_ input: AnyCommandEncoder<MPSImage>) -> AnyCommandEncoder<MPSImage> {
        self.input = input
        input.registerConsumer()
        return AnyCommandEncoder<MPSImage>(self)
    }
    
    func registerConsumer() {
        consumerCount += 1
    }
    
    func forward(commandBuffer: MTLCommandBuffer) -> MPSImage {
        if outputMemoId != nil && outputMemoId! == commandBuffer.hash {
            return outputMemo!
        } else {
            let sourceImage = input.forward(commandBuffer: commandBuffer)
            let destinationImage = self.destinationImage(sourceImage: sourceImage, commandBuffer: commandBuffer)
            let biasesConverted = Conversions.float32toFloat16(pointer: UnsafeMutableRawPointer(mutating: self.biases.pointer), count: self.biases.count)
            let biasesBuffer = ShaderRegistry.getDevice().makeBuffer(
                bytes: biasesConverted,
                length: biasesConverted.count * ExpectedFloat16Size,
                options: MTLResourceOptions.cpuCacheModeWriteCombined)
            encode(commandBuffer: commandBuffer, biases: biasesBuffer, sourceImage: sourceImage, destinationImage: destinationImage)
            outputMemoId = commandBuffer.hash
            outputMemo = destinationImage
            return outputMemo!
        }
    }
    
    private func encode(commandBuffer: MTLCommandBuffer, biases: MTLBuffer, sourceImage: MPSImage, destinationImage: MPSImage) {
        let bnPipelineState = ShaderRegistry.getOrLoad(name: "add_bias")
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.setComputePipelineState(bnPipelineState)
        encoder.setTexture(sourceImage.texture, at: 0)
        encoder.setTexture(destinationImage.texture, at: 1)
        encoder.setBuffer(biases, offset: 0, at: 2)
        let threadGroupWidth = bnPipelineState.threadExecutionWidth
        let threadGroupHeight = bnPipelineState.maxTotalThreadsPerThreadgroup / threadGroupWidth
        let threadGroupShape = MTLSizeMake(threadGroupWidth, threadGroupHeight, 1)
        let gridShape = MTLSize(
            width: (destinationImage.width + threadGroupWidth - 1) / threadGroupWidth,
            height: (destinationImage.height + threadGroupHeight - 1) / threadGroupHeight,
            depth: destinationImage.texture.arrayLength)
        encoder.dispatchThreadgroups(gridShape, threadsPerThreadgroup: threadGroupShape)
        encoder.endEncoding()
        
        if let image = sourceImage as? MPSTemporaryImage {
            image.readCount -= 1
        }
    }
    
    private func destinationImage(sourceImage: MPSImage, commandBuffer: MTLCommandBuffer) -> MPSImage {
        let textureDesc = MTLTextureDescriptor()
        textureDesc.arrayLength = sourceImage.texture.arrayLength
        textureDesc.height = sourceImage.height
        textureDesc.width = sourceImage.width
        textureDesc.textureType = .type2DArray
        textureDesc.usage = MTLTextureUsage(rawValue:
            MTLTextureUsage.shaderWrite.rawValue | MTLTextureUsage.shaderRead.rawValue)
        textureDesc.pixelFormat = .rgba16Float
        
        if useTemporary {
            let img = MPSTemporaryImage.init(commandBuffer: commandBuffer, textureDescriptor: textureDesc)
            img.readCount = consumerCount
            return img
        } else {
            let texture = ShaderRegistry.getDevice().makeTexture(descriptor: textureDesc)
            return MPSImage.init(texture: texture, featureChannels: max(4, sourceImage.featureChannels))
        }
    }
}
