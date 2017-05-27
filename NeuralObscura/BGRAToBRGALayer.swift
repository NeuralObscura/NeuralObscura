//
//  BGRAToBRGALayer.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 11/26/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class BGRAToBRGALayer: UnaryCommandEncoder {
    private let useTemporary: Bool
    private var consumerCount: Int = 0
    var input: AnyCommandEncoder<MPSImage>!
    private var outputMemoId: Int?
    private var outputMemo: MPSImage?
    
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
        let textureDesc = MTLTextureDescriptor()
        textureDesc.height = sourceImage.height
        textureDesc.width = sourceImage.width
        textureDesc.textureType = .type2D
        textureDesc.usage = MTLTextureUsage(rawValue:
            MTLTextureUsage.shaderWrite.rawValue | MTLTextureUsage.shaderRead.rawValue)
        textureDesc.pixelFormat = .rgba16Float
        
        if useTemporary {
            let img = MPSTemporaryImage.init(commandBuffer: commandBuffer, textureDescriptor: textureDesc)
            img.readCount = consumerCount
            return img
        } else {
            let texture = ShaderRegistry.getDevice().makeTexture(descriptor: textureDesc)
            return MPSImage(texture: texture, featureChannels: max(4, sourceImage.featureChannels))
        }

    }
    
    func forward(commandBuffer: MTLCommandBuffer) -> MPSImage {
        if outputMemoId != nil && outputMemoId! == commandBuffer.hash {
            return outputMemo!
        } else {
            let sourceImage = input.forward(commandBuffer: commandBuffer)
            let destinationImage = self.destinationImage(sourceImage: sourceImage, commandBuffer: commandBuffer)
            let encoder = commandBuffer.makeComputeCommandEncoder()
            let pipelineState = ShaderRegistry.getOrLoad(name: "bgra_to_brga")
            encoder.setComputePipelineState(pipelineState)
            encoder.setTexture(sourceImage.texture, at: 0)
            encoder.setTexture(destinationImage.texture, at: 1)
            
            let threadGroupWidth = pipelineState.threadExecutionWidth
            let threadGroupHeight = pipelineState.maxTotalThreadsPerThreadgroup / threadGroupWidth
            let threadGroupShape = MTLSizeMake(threadGroupWidth, threadGroupHeight, 1)
            let gridShape = MTLSize(
                width: (destinationImage.width + threadGroupWidth - 1) / threadGroupWidth,
                height: (destinationImage.height + threadGroupHeight - 1) / threadGroupHeight,
                depth: destinationImage.texture.arrayLength)
            
            encoder.dispatchThreadgroups(gridShape, threadsPerThreadgroup: threadGroupShape)
            encoder.endEncoding()
            
            if sourceImage is MPSTemporaryImage {
                (sourceImage as! MPSTemporaryImage).readCount -= 1
            }
            outputMemoId = commandBuffer.hash
            outputMemo = destinationImage
            return outputMemo!
        }
    }
}
