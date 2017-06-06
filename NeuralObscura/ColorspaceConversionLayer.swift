//
//  ColorspaceConversionLayer.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 6/5/17.
//  Copyright © 2017 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class ColorspaceConversionLayer: UnaryCommandEncoder {
    private let useTemporary: Bool
    private var consumerCount: Int = 0
    private let imageConversion: MPSImageConversion
    private var input: AnyCommandEncoder<MPSImage>!
    private var outputMemoId: Int?
    private var outputMemo: MPSImage?

    private let sourceColorSpace: CGColorSpace
    private let destinationColorSpace: CGColorSpace

    init(
        sourceColorSpace: CGColorSpace,
        destinationColorSpace: CGColorSpace,
        useTemporary: Bool = false) {

        self.useTemporary = useTemporary
        self.sourceColorSpace = sourceColorSpace
        self.destinationColorSpace = destinationColorSpace

        let conversionInfo = CGColorConversionInfo(src: sourceColorSpace,
                                                   dst: destinationColorSpace)
        imageConversion = MPSImageConversion(device: ShaderRegistry.getDevice(),
                                             srcAlpha: .alphaIsOne,
                                             destAlpha: .alphaIsOne,
                                             backgroundColor: nil,
                                             conversionInfo: conversionInfo)
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
            imageConversion.encode(commandBuffer: commandBuffer,
                                   sourceTexture: sourceImage.texture,
                                   destinationTexture: destinationImage.texture)
            outputMemoId = commandBuffer.hash
            outputMemo = destinationImage
            return outputMemo!
        }
    }

    private func destinationImage(sourceImage: MPSImage, commandBuffer: MTLCommandBuffer) -> MPSImage {
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
            return MPSImage.init(texture: texture, featureChannels: 3)
        }
    }
}
