//
//  Col2ImLayer.swift
//  NeuralObscura
//
//  Created by Edward Knox on 3/19/17.
//  Copyright © 2017 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class Col2ImLayer: BinaryCommandEncoder {
    private var inputA: AnyCommandEncoder<MTLBuffer>!
    private var inputB: AnyCommandEncoder<MPSImage>!
    private var outputMemoId: Int?
    private var outputMemo: MPSImage?
    private let useTemporary: Bool
    private var consumerCount: Int = 0
    
    private let channelsOut: UInt32
    private let kernelSize: UInt32
    private let stride: UInt32
    private let padding: UInt32
    
    // TODO: Refactor this constructor
    init(channelsOut: UInt32,
         kernelSize: UInt32,
         stride: UInt32,
         padding: UInt32,
         useTemporary: Bool = false) {
        self.useTemporary = useTemporary
        self.channelsOut = channelsOut
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
    }
    
    func chain(
        _ inputA: AnyCommandEncoder<MTLBuffer>,
        _ inputB: AnyCommandEncoder<MPSImage>) -> AnyCommandEncoder<MPSImage> {
        self.inputA = inputA
        self.inputB = inputB
        inputA.registerConsumer()
        inputB.registerConsumer()
        return AnyCommandEncoder<MPSImage>(self)
    }
    
    func forward(commandBuffer: MTLCommandBuffer) -> MPSImage {
        if outputMemoId != nil && outputMemoId! == commandBuffer.hash {
            return outputMemo!
        } else {
            let sourceBuffer = inputA.forward(commandBuffer: commandBuffer)
            let originalImage = inputB.forward(commandBuffer: commandBuffer)
            let destImage: MPSImage = destinationImage(sourceImage: originalImage, commandBuffer: commandBuffer)
            
            // TODO: Configure this with constructor parameters
            let nh = UInt32(originalImage.height)
            let nw = UInt32(originalImage.width)

            let params = [
                channelsOut,
                nh,
                nw,
                kernelSize,
                kernelSize,
                stride,
                padding] as [UInt32]
            
            let paramsBuffer = ShaderRegistry.getDevice().makeBuffer(
                bytes: params,
                length: params.count * MemoryLayout<UInt32>.size,
                options: .cpuCacheModeWriteCombined)
            
            let encoder = commandBuffer.makeComputeCommandEncoder()
            let state = ShaderRegistry.getOrLoad(name: "col2im")
            encoder.setComputePipelineState(state)
            encoder.setBuffer(sourceBuffer, offset: 0, at: 0)
            encoder.setTexture(destImage.texture, at: 1)
            encoder.setBuffer(paramsBuffer, offset:0, at: 2)
            let threadsPerGroup = MTLSizeMake(1, 1, 1)
            // TODO: Set thread group size! Not sure how bytes figures into this
            // TODO: double check this
            let threadGroups = MTLSizeMake(1, 1, 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
            encoder.endEncoding()
            
            outputMemoId = commandBuffer.hash
            outputMemo = destImage
            return outputMemo!
        }
    }
    
    func registerConsumer() {
        consumerCount += 1
    }
    
    func destinationImage(sourceImage: MPSImage, commandBuffer: MTLCommandBuffer) -> MPSImage {
        let inHeight = UInt32(sourceImage.height)
        let inWidth = UInt32(sourceImage.width)
        
        let stride = self.stride
        
        let outHeight = stride * (inHeight - 1) + kernelSize - 2 * padding
        let outWidth = stride * (inWidth - 1) + kernelSize - 2 * padding
        
        /* Assert the constraint on input size, kernel size, padding, stride. */
        assert((outHeight + 2 * padding - kernelSize) % stride == 0,
               "Input size must be a multiple of i+2p-k in all dimensions. This constraint is failing in the height dimension.")
        assert((outWidth + 2 * padding - kernelSize) % stride == 0,
               "Input size must be a multiple of i+2p-k in all dimensions. This constraint is failing in the width dimension.")
        
        let textureDesc = MTLTextureDescriptor()
        textureDesc.arrayLength = (Int(channelsOut) + 3) / 4
        textureDesc.height = Int(outHeight)
        textureDesc.width = Int(outWidth)
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
            return MPSImage.init(texture: texture, featureChannels: max(4, Int(channelsOut)))
        }
    }
}
