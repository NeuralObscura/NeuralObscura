//
//  TanhAdjustmentLayer.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 11/6/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class TanhAdjustmentLayer: UnaryCommandEncoder {
    private let useTemporary: Bool
    private var consumerCount: Int = 0
    var input: AnyCommandEncoder<MPSImage>!
    
    init(useTemporary: Bool = false) {
        self.useTemporary = useTemporary
    }
    
    func chain(_ input: AnyCommandEncoder<MPSImage>) -> AnyCommandEncoder<MPSImage> {
        self.input = input
        input.registerConsumer()
        return AnyCommandEncoder<MPSImage>(self)
    }
    
    func registerConsumer() {
        consumerCount += 1
    }
    
    func destinationImage(sourceImage: MPSImage, commandBuffer: MTLCommandBuffer) -> MPSImage {
        let destDesc = MPSImageDescriptor(
            channelFormat: textureFormat,
            width: sourceImage.width,
            height: sourceImage.height,
            featureChannels: sourceImage.featureChannels)
        if useTemporary {
            let img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: destDesc)
            img.readCount = consumerCount
            return img
        } else {
            return MPSImage(device: ShaderRegistry.getDevice(), imageDescriptor: destDesc)
        }
    }
    
    func forward(commandBuffer: MTLCommandBuffer) -> MPSImage {
        let sourceImage = input.forward(commandBuffer: commandBuffer)
        let destinationImage = self.destinationImage(sourceImage: sourceImage, commandBuffer: commandBuffer)
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.setComputePipelineState(ShaderRegistry.getOrLoad(name: "tanh_adjustment"))
        encoder.setTexture(sourceImage.texture, at: 0)
        encoder.setTexture(destinationImage.texture, at: 1)
        let threadsPerGroup = MTLSizeMake(1, 1, 1)
        let threadGroups = MTLSizeMake(destinationImage.texture.width,
                                       destinationImage.texture.height,
                                       destinationImage.texture.arrayLength)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        if sourceImage is MPSTemporaryImage {
            (sourceImage as! MPSTemporaryImage).readCount -= 1
        }
        return destinationImage
    }
}
