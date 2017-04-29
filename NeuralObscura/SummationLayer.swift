//
//  SummationLayer.swift
//  NeuralObscura
//
//  Created by Edward Knox on 11/14/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class SummationLayer: BinaryCommandEncoder {
    typealias InputTypeA = MPSImage
    typealias InputTypeB = MPSImage

    private var inputsSupplied = 0
    private let useTemporary: Bool
    private var consumerCount: Int = 0
    var inputA: AnyCommandEncoder<InputTypeA>!
    var inputB: AnyCommandEncoder<InputTypeB>!
    /* Int should be a mtlCommandBuffer.hash value */
    private var outputMemoId: Int?
    private var outputMemo: MPSImage?
    
    init(useTemporary: Bool = false) {
        self.useTemporary = useTemporary
    }
    
    func chain(
        _ inputA: AnyCommandEncoder<InputTypeA>,
        _ inputB: AnyCommandEncoder<InputTypeB>) -> AnyCommandEncoder<MPSImage> {
        self.inputA = inputA
        self.inputB = inputB
        inputA.registerConsumer()
        return AnyCommandEncoder<MPSImage>(self)
    }
    
    func registerConsumer() {
        consumerCount += 1
    }
    
    func forward(commandBuffer: MTLCommandBuffer) -> MPSImage {
        if outputMemoId != nil && outputMemoId! == commandBuffer.hash {
            return outputMemo!
        } else {
            let sourceImageA = inputA.forward(commandBuffer: commandBuffer)
            let sourceImageB = inputB.forward(commandBuffer: commandBuffer)
            let destinationImage = self.destinationImage(sourceImageA: sourceImageA, commandBuffer: commandBuffer)
            encode(
                commandBuffer: commandBuffer,
                sourceImageA: sourceImageA,
                sourceImageB: sourceImageB,
                destinationImage: destinationImage)
            outputMemoId = commandBuffer.hash
            outputMemo = destinationImage
            return outputMemo!
        }
    }
    
    private func destinationImage(sourceImageA: MPSImage, commandBuffer: MTLCommandBuffer) -> MPSImage {
        let textureDesc = MTLTextureDescriptor()
        textureDesc.arrayLength = sourceImageA.texture.arrayLength
        textureDesc.height = sourceImageA.height
        textureDesc.width = sourceImageA.width
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
            return MPSImage.init(texture: texture, featureChannels: max(4, sourceImageA.featureChannels))
        }
    }
    
    private func encode(
        commandBuffer: MTLCommandBuffer,
        sourceImageA: MPSImage,
        sourceImageB: MPSImage,
        destinationImage: MPSImage) {
        
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.setComputePipelineState(ShaderRegistry.getOrLoad(name: "add"))
        encoder.setTexture(sourceImageA.texture, at: 0)
        encoder.setTexture(sourceImageB.texture, at: 1)
        encoder.setTexture(destinationImage.texture, at: 2)
        let threadsPerGroup = MTLSizeMake(1, 1, 1)
        let threadGroups = MTLSizeMake(destinationImage.texture.width,
                                       destinationImage.texture.height,
                                       destinationImage.texture.arrayLength)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        
        if let image = sourceImageA as? MPSTemporaryImage {
            image.readCount -= 1
        }
        if let image = sourceImageB as? MPSTemporaryImage {
            image.readCount -= 1
        }
        inputsSupplied = 0
    }
}
