//
//  BatchNormalizationLayer.swift
//  NeuralObscura
//
//  Created by Edward Knox on 9/27/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class BatchNormalizationNonTestLayer: UnaryCommandEncoder {
    private let beta: ParameterBuffer
    private let gamma: ParameterBuffer
    private let useTemporary: Bool
    private var consumerCount: Int = 0
    private var input: AnyCommandEncoder<MPSImage>!
    private var outputMemoId: Int?
    private var outputMemo: MPSImage?
    
    init(beta: ParameterBuffer,
         gamma: ParameterBuffer,
         useTemporary: Bool = false) {
        self.useTemporary = useTemporary
        self.beta = beta
        self.gamma = gamma
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
            encode(commandBuffer: commandBuffer, sourceImage: sourceImage, destinationImage: destinationImage)
            outputMemoId = commandBuffer.hash
            outputMemo = destinationImage
            return outputMemo!
        }
    }
    
    private func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage) {
        let meanBuf = ShaderRegistry.getDevice().makeBuffer(
            length: MemoryLayout<Float32>.size * sourceImage.texture.arrayLength * 4,
            options: MTLResourceOptions.storageModeShared)
        let stddevBuf = ShaderRegistry.getDevice().makeBuffer(
            length: MemoryLayout<Float32>.size * sourceImage.texture.arrayLength * 4,
            options: MTLResourceOptions.storageModeShared)
        let meanVarEnc = commandBuffer.makeComputeCommandEncoder()
        meanVarEnc.setComputePipelineState(ShaderRegistry.getOrLoad(name: "mean_std_dev"))
        meanVarEnc.setTexture(sourceImage.texture, at: 0)
        meanVarEnc.setBuffer(meanBuf, offset: 0, at: 1)
        meanVarEnc.setBuffer(stddevBuf, offset: 0, at: 2)
        let threadsPerGroupMeanVar = MTLSizeMake(1, 1, 1)
        let threadGroupsMeanVar = MTLSizeMake(1, 1, sourceImage.texture.arrayLength)
        meanVarEnc.dispatchThreadgroups(threadGroupsMeanVar, threadsPerThreadgroup: threadsPerGroupMeanVar)
        meanVarEnc.endEncoding()
        
        let betaBuf = ShaderRegistry.getDevice()
            .makeBuffer(
                bytes: beta.pointer,
                length: beta.length,
                options: MTLResourceOptions.cpuCacheModeWriteCombined)
        let gammaBuf = ShaderRegistry.getDevice()
            .makeBuffer(
                bytes: gamma.pointer,
                length: gamma.length,
                options: MTLResourceOptions.cpuCacheModeWriteCombined)
        let bnEnc = commandBuffer.makeComputeCommandEncoder()
        bnEnc.setComputePipelineState(ShaderRegistry.getOrLoad(name: "batch_normalization_nt"))
        bnEnc.setTexture(sourceImage.texture, at: 0)
        bnEnc.setTexture(destinationImage.texture, at: 1)
        bnEnc.setBuffer(gammaBuf, offset: 0, at: 2)
        bnEnc.setBuffer(betaBuf, offset: 0, at: 3)
        bnEnc.setBuffer(meanBuf, offset: 0, at: 4)
        bnEnc.setBuffer(stddevBuf, offset: 0, at: 5)
        let threadsPerGroup = MTLSizeMake(1, 1, 1)
        let threadGroups = MTLSizeMake(1, 1, sourceImage.texture.arrayLength)
        bnEnc.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        bnEnc.endEncoding()
        
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

