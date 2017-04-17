//
//  ConvolutionLayer.swift
//  NeuralObscura
//
//  Created by Edward Knox on 9/25/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class ConvolutionLayer: UnaryCommandEncoder {
    private let useTemporary: Bool
    private var consumerCount: Int = 0
    private let convolution: MPSCNNConvolution
    private let padding: Int
    private var input: AnyCommandEncoder<MPSImage>!
    private var outputMemoId: Int?
    private var outputMemo: MPSImage?
    
    init(
        kernelSize: Int,
        channelsIn: Int,
        channelsOut: Int,
        w: ParameterBuffer,
        b: ParameterBuffer,
        relu: Bool = true,
        padding: Int = 0,
        stride: Int = 1,
        destinationFeatureChannelOffset: Int = 0,
        groupNum: Int = 1,
        useTemporary: Bool = false) {
        
        self.useTemporary = useTemporary
        self.padding = padding
        
        var neuronFilter: MPSCNNNeuron?
        if relu {
            neuronFilter = MPSCNNNeuronReLU(device: ShaderRegistry.getDevice(), a: 0)
        }
        
        // create appropriate convolution descriptor with appropriate stride
        let convDesc = MPSCNNConvolutionDescriptor(kernelWidth: Int(kernelSize),
                                                   kernelHeight: Int(kernelSize),
                                                   inputFeatureChannels: Int(channelsIn),
                                                   outputFeatureChannels: Int(channelsOut),
                                                   neuronFilter: neuronFilter)
        convDesc.strideInPixelsX = stride
        convDesc.strideInPixelsY = stride
        
        assert((groupNum > 0), "Group size can't be less than 1")
        convDesc.groups = Int(groupNum)
        
        // initialize the convolution layer by calling the parent's (MPSCNNConvlution's) initializer
        convolution = MPSCNNConvolution(
            device: ShaderRegistry.getDevice(),
            convolutionDescriptor: convDesc,
            kernelWeights: w.pointer(),
            biasTerms: b.pointer(),
            flags: MPSCNNConvolutionFlags.none)
        convolution.destinationFeatureChannelOffset = Int(destinationFeatureChannelOffset)
        convolution.edgeMode = .zero
        convolution.offset = MPSOffset(x: (convolution.kernelWidth / 2) - padding, y: (convolution.kernelHeight / 2) - padding, z: 0)
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
            convolution.encode(commandBuffer: commandBuffer, sourceImage: sourceImage, destinationImage: destinationImage)
            outputMemoId = commandBuffer.hash
            outputMemo = destinationImage
            return outputMemo!
        }
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
        let inHeight = sourceImage.height
        let inWidth = sourceImage.width
        
        let kernelSize = convolution.kernelWidth
        let stride = convolution.strideInPixelsX
        let channelsOut = convolution.outputFeatureChannels

        let outHeight = ((inHeight + (2 * self.padding) - kernelSize) / stride) + 1
        let outWidth = ((inWidth + (2 * self.padding) - kernelSize) / stride) + 1
        let descriptor = MPSImageDescriptor(
            channelFormat: textureFormat,
            width: outWidth,
            height: outHeight,
            featureChannels: channelsOut)

        return descriptor
    }
}
