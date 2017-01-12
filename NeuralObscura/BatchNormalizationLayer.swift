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
    init(
        channelsIn: Int,
        beta: ParameterBuffer,
        gamma: ParameterBuffer,
        mean: ParameterBuffer? = nil,
        stddev: ParameterBuffer? = nil,
        testMode: Bool = false,
        debug: Bool = false) {
        super.init(
            delegate: BatchNormalizationLayerDelegate(
                channelsIn: channelsIn,
                beta: beta,
                gamma: gamma,
                mean: mean,
                stddev: stddev,
                testMode: testMode),
            debug: debug)
    }
}

class BatchNormalizationLayerDelegate: CommandEncoderDelegate {
    let beta: MTLBuffer
    let gamma: MTLBuffer
    let mean: MTLBuffer
    let stddev: MTLBuffer
    let channelsIn: Int
    let testMode: Bool
    
    private var sourceImage: MPSImage!

    init(
         channelsIn: Int,
         beta: ParameterBuffer,
         gamma: ParameterBuffer,
         mean: ParameterBuffer? = nil,
         stddev: ParameterBuffer? = nil,
         testMode: Bool) {
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
    
    func getDestinationImageDescriptor(sourceImage: MPSImage) -> MPSImageDescriptor {
        return MPSImageDescriptor(
            channelFormat: textureFormat,
            width: sourceImage.width,
            height: sourceImage.height,
            featureChannels: sourceImage.featureChannels)
    }
    
    func supplyInput(sourceImage: MPSImage, sourcePosition: Int) -> Bool {
        self.sourceImage = sourceImage
        return true
    }
    
    func encode(commandBuffer: MTLCommandBuffer, destinationImage: MPSImage) {
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
