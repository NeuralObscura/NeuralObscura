//
//  FullyConnectedLayer.swift
//  NeuralObscura
//
//  Created by Edward Knox on 9/25/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class FullyConnectedLayer: UnaryCommandEncoder {
    private let fullyConnected: MPSCNNFullyConnected
    private let useTemporary: Bool
    private var consumerCount: Int = 0
    private var input: AnyCommandEncoder<MPSImage>!
    private var outputMemoId: Int?
    private var outputMemo: MPSImage?
    
    init(
        kernelSize: Int,
        channelsIn: Int,
        channelsOut: Int,
        w: ParameterBuffer,
        b: ParameterBuffer,
        neuronFilter: MPSCNNNeuron? = MPSCNNNeuronSigmoid(),
        destinationFeatureChannelOffset: Int = 0,
        useTemporary: Bool = false) {
        
        self.useTemporary = useTemporary
        // create appropriate convolution descriptor (in fully connected, stride is always 1)
        let convDesc = MPSCNNConvolutionDescriptor(kernelWidth: kernelSize,
                                                   kernelHeight: kernelSize,
                                                   inputFeatureChannels: channelsIn,
                                                   outputFeatureChannels: channelsOut,
                                                   neuronFilter: neuronFilter)
        
        // initialize the convolution layer by calling the parent's (MPSCNNFullyConnected's) initializer
        fullyConnected = MPSCNNFullyConnected.init(device: ShaderRegistry.getDevice(),
                   convolutionDescriptor: convDesc,
                   kernelWeights: w.pointer,
                   biasTerms: b.pointer,
                   flags: MPSCNNConvolutionFlags.none)
        fullyConnected.destinationFeatureChannelOffset = destinationFeatureChannelOffset
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
            let destinationImage = self.destinationImage(input: sourceImage, commandBuffer: commandBuffer)
            fullyConnected.encode(commandBuffer: commandBuffer, sourceImage: sourceImage, destinationImage: destinationImage)
            outputMemoId = commandBuffer.hash
            outputMemo = destinationImage
            return outputMemo!
        }
    }
    
    private func destinationImage(input: MPSImage, commandBuffer: MTLCommandBuffer) -> MPSImage {
        let textureDesc = MTLTextureDescriptor()
        textureDesc.arrayLength = ((fullyConnected.outputFeatureChannels + 3) / 4)
        textureDesc.height = 1
        textureDesc.width = 1
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
            return MPSImage.init(texture: texture, featureChannels: max(4, fullyConnected.outputFeatureChannels))
        }
    }
}
