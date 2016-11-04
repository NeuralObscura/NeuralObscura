//
//  BatchNormalizationLayer.swift
//  NeuralObscura
//
//  Created by Edward Knox on 9/27/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class BatchNormalizationLayer: CommandEncoder {
    init(
        device: MTLDevice,
        shader: MTLComputePipelineState,
        channelsIn: UInt,
        beta: ParameterBuffer,
        gamma: ParameterBuffer,
        outputType: CommandEncoderOutputType = CommandEncoderOutputType.debug) {
        super.init(
            device: device,
            delegate: BatchNormalizationLayerDelegate(
                device: device,
                shader: shader,
                channelsIn: channelsIn,
                beta: beta,
                gamma: gamma),
            outputType: outputType)
    }
}

class BatchNormalizationLayerDelegate: CommandEncoderDelegate {
    let beta: MTLBuffer
    let gamma: MTLBuffer
    let channelsIn: Int
    let shader: MTLComputePipelineState

    init(device: MTLDevice,
         shader: MTLComputePipelineState,
         channelsIn: UInt,
         beta: ParameterBuffer,
         gamma: ParameterBuffer) {
        self.channelsIn = Int(channelsIn)
        self.beta = device.makeBuffer(bytes: beta.pointer(), length: beta.lengthInBytes(), options: MTLResourceOptions.cpuCacheModeWriteCombined)
        self.gamma = device.makeBuffer(bytes: gamma.pointer(), length: gamma.lengthInBytes(), options: MTLResourceOptions.cpuCacheModeWriteCombined)
        self.shader = shader
    }
    
    func getDestinationImageDescriptor(sourceImage: MPSImage) -> MPSImageDescriptor {
        return MPSImageDescriptor(
            channelFormat: textureFormat,
            width: sourceImage.width,
            height: sourceImage.height,
            featureChannels: sourceImage.featureChannels)
    }
    
    func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage) {
        print("bn encode")
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.setComputePipelineState(shader)
        encoder.setTexture(sourceImage.texture, at: 0)
        encoder.setTexture(destinationImage.texture, at: 1)
        encoder.setBuffer(gamma, offset: 0, at: 2)
        encoder.setBuffer(beta, offset: 0, at: 3)
        let threadsPerGroup = MTLSizeMake(1, 1, 1)
        let threadGroups = MTLSizeMake(destinationImage.texture.width / threadsPerGroup.width,
                                       destinationImage.texture.height / threadsPerGroup.height, 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        if sourceImage is MPSTemporaryImage {
            (sourceImage as! MPSTemporaryImage).readCount -= 1
        }
    }
}
