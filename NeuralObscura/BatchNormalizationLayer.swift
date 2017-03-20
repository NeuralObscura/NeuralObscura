//
//  BatchNormalizationLayer.swift
//  NeuralObscura
//
//  Created by Edward Knox on 9/27/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class BatchNormalizationLayer: UnaryCommandEncoder {
    private let beta: MTLBuffer
    private let gamma: MTLBuffer
    private let mean: MTLBuffer
    private let stddev: MTLBuffer
    private let channelsIn: Int
    private let testMode: Bool
    private let useTemporary: Bool
    private var consumerCount: Int = 0
    private var input: AnyCommandEncoder<MPSImage>!
    
    init(
        channelsIn: Int,
        beta: ParameterBuffer,
        gamma: ParameterBuffer,
        mean: ParameterBuffer? = nil,
        stddev: ParameterBuffer? = nil,
        testMode: Bool = false,
        useTemporary: Bool = false) {
        
        self.useTemporary = useTemporary
        self.channelsIn = Int(channelsIn)
        self.testMode = testMode
        self.beta = ShaderRegistry.getDevice().makeBuffer(bytes: beta.pointer(), length: beta.lengthInBytes(), options: MTLResourceOptions.cpuCacheModeWriteCombined)
        self.gamma = ShaderRegistry.getDevice().makeBuffer(bytes: gamma.pointer(), length: gamma.lengthInBytes(), options: MTLResourceOptions.cpuCacheModeWriteCombined)
        if self.testMode {
            self.mean = ShaderRegistry.getDevice().makeBuffer(bytes: mean!.pointer(), length: mean!.lengthInBytes(), options: MTLResourceOptions.cpuCacheModeWriteCombined)
            self.stddev = ShaderRegistry.getDevice().makeBuffer(bytes: stddev!.pointer(), length: stddev!.lengthInBytes(), options: MTLResourceOptions.cpuCacheModeWriteCombined)
        } else {
            self.mean = ShaderRegistry.getDevice().makeBuffer(length: channelsIn * MemoryLayout<Float32>.size, options: MTLResourceOptions.storageModePrivate)
            self.stddev = ShaderRegistry.getDevice().makeBuffer(length: channelsIn * MemoryLayout<Float32>.size, options: MTLResourceOptions.storageModePrivate)
        }
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
        let sourceImage = input.forward(commandBuffer: commandBuffer)
        let destinationImage = self.destinationImage(sourceImage: sourceImage, commandBuffer: commandBuffer)
        encode(commandBuffer: commandBuffer, sourceImage: sourceImage, destinationImage: destinationImage)
        return destinationImage
    }
    
    private func destinationImage(sourceImage: MPSImage, commandBuffer: MTLCommandBuffer) -> MPSImage {
        let destDesc = destinationImageDescriptor(sourceImage: sourceImage)
        if useTemporary {
            let img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: destDesc)
            img.readCount = consumerCount
            return img
        } else {
            return MPSImage(device: ShaderRegistry.getDevice(), imageDescriptor: destDesc)
        }
    }
    
    private func destinationImageDescriptor(sourceImage: MPSImage) -> MPSImageDescriptor {
        return MPSImageDescriptor(
            channelFormat: textureFormat,
            width: sourceImage.width,
            height: sourceImage.height,
            featureChannels: sourceImage.featureChannels)
    }
    
    private func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage) {
        let encoder = commandBuffer.makeComputeCommandEncoder()
        if testMode {
            encoder.setComputePipelineState(ShaderRegistry.getOrLoad(name: "batch_normalization"))
            encoder.setTexture(sourceImage.texture, at: 0)
            encoder.setTexture(destinationImage.texture, at: 1)
            encoder.setBuffer(gamma, offset: 0, at: 2)
            encoder.setBuffer(beta, offset: 0, at: 3)
            encoder.setBuffer(mean, offset: 0, at: 4)
            encoder.setBuffer(stddev, offset: 0, at: 5)
            let threadsPerGroup = MTLSizeMake(1, 1, 1)
            let threadGroups = MTLSizeMake(sourceImage.texture.width,
                                           sourceImage.texture.height,
                                           sourceImage.texture.arrayLength)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        } else {
            encoder.setComputePipelineState(ShaderRegistry.getOrLoad(name: "batch_normalization_nt"))
            encoder.setTexture(sourceImage.texture, at: 0)
            encoder.setTexture(destinationImage.texture, at: 1)
            encoder.setBuffer(gamma, offset: 0, at: 2)
            encoder.setBuffer(beta, offset: 0, at: 3)
            let threadsPerGroup = MTLSizeMake(1, 1, 1)
            let threadGroups = MTLSizeMake(1, 1,
                                            channelsIn / sourceImage.texture.pixelFormat.channelCount)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)

        }
        encoder.endEncoding()

        if let image = sourceImage as? MPSTemporaryImage {
            image.readCount -= 1
        }
    }
}
