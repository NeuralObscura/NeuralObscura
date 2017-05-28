//
// Created by Edward Knox on 5/22/17.
// Copyright (c) 2017 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

/**
 Computes a deconvolution operation.
 
 Our reference implementation found at: https://arxiv.org/pdf/1603.07285.pdf
 
 Our implementation applies the constraint where in both dimensions the input size is a multiple
 of `i + 2p - k` where `i` is the input size in that dimension, `p` is the padding in that dimension,
 and `k` is the kernel size in that dimension.
 */
class DeconvolutionLayerV2: UnaryCommandEncoder {
    private let useTemporary: Bool
    private var consumerCount: Int = 0
    private var input: AnyCommandEncoder<MPSImage>!
    private var outputMemoId: Int?
    private var outputMemo: MPSImage?
    
    private let kernelSize: Int
    private let padding: Int
    private let stride: Int
    private let biases: ParameterBuffer
    private let channelsIn: Int
    private let channelsOut: Int
    
    private let transposedWeights: ParameterBuffer
    private let transposedPadding: Int
    
    private let convolution: MPSCNNConvolution!
    
    init(kernelSize: Int,
         channelsIn: Int,
         channelsOut: Int,
         weights: ParameterBuffer,
         biases: ParameterBuffer,
         relu: Bool = true,
         padding: Int = 0,
         stride: Int = 1,
         useTemporary: Bool = false) {
        
        self.useTemporary = useTemporary
        
        self.padding = padding
        self.kernelSize = kernelSize
        self.biases = biases
        self.channelsIn = channelsIn
        self.channelsOut = channelsOut
        self.stride = stride
        
        self.transposedPadding = kernelSize - padding - 1
        self.transposedWeights = DeconvolutionLayerV2.transformWeights(
            weights: weights,
            channelsOut: channelsOut,
            kernelSize: kernelSize,
            channelsIn: channelsIn)
        
        var neuronFilter: MPSCNNNeuron?
        if relu {
            neuronFilter = MPSCNNNeuronReLU(device: ShaderRegistry.getDevice(), a: 0)
        }
        
        // create appropriate convolution descriptor with appropriate stride
        let convDesc = MPSCNNConvolutionDescriptor(
            kernelWidth: kernelSize,
            kernelHeight: kernelSize,
            inputFeatureChannels: channelsIn,
            outputFeatureChannels: channelsOut,
            neuronFilter: neuronFilter)
        convDesc.strideInPixelsX = 1
        convDesc.strideInPixelsY = 1
        convDesc.groups = 1
        
        // initialize the convolution layer by calling the parent's (MPSCNNConvlution's) initializer
        self.convolution = MPSCNNConvolution(
            device: ShaderRegistry.getDevice(),
            convolutionDescriptor: convDesc,
            kernelWeights: self.transposedWeights.pointer,
            biasTerms: self.biases.pointer,
            flags: MPSCNNConvolutionFlags.none)
        self.convolution.destinationFeatureChannelOffset = 0
        self.convolution.edgeMode = .zero
        self.convolution.offset = MPSOffset(
            x: (kernelSize / 2) - transposedPadding,
            y: (kernelSize / 2) - transposedPadding,
            z: 0)
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
            var intermediateImage = sourceImage
            
            if stride > 1 {
                let textureDesc = MTLTextureDescriptor()
                textureDesc.textureType = .type2DArray
                textureDesc.width = (sourceImage.width - 1) * stride + 1
                textureDesc.height = (sourceImage.height - 1) * stride + 1
                textureDesc.pixelFormat = .rgba16Float
                textureDesc.arrayLength = sourceImage.texture.arrayLength
                let texture = ShaderRegistry.getDevice().makeTexture(descriptor: textureDesc)
                let intermediateImage = MPSImage(texture: texture, featureChannels: sourceImage.featureChannels)
                
                var ustride = UInt32(stride)
                let strideBuf = ShaderRegistry.getDevice().makeBuffer(
                    bytes: &ustride,
                    length: MemoryLayout<UInt32>.stride,
                    options: MTLResourceOptions.cpuCacheModeWriteCombined)
                let encoder = commandBuffer.makeComputeCommandEncoder()
                let pipelineState = ShaderRegistry.getOrLoad(name: "deconvolution_interpixel_stride")
                encoder.setComputePipelineState(pipelineState)
                encoder.setTexture(sourceImage.texture, at: 0)
                encoder.setTexture(intermediateImage.texture, at: 1)
                encoder.setBuffer(strideBuf, offset: 0, at: 2)
                let threadGroupWidth = pipelineState.threadExecutionWidth
                let threadGroupHeight = pipelineState.maxTotalThreadsPerThreadgroup / threadGroupWidth
                let threadGroupShape = MTLSizeMake(threadGroupWidth, threadGroupHeight, 1)
                let gridShape = MTLSize(
                    width: (sourceImage.width + threadGroupWidth - 1) / threadGroupWidth,
                    height: (sourceImage.height + threadGroupHeight - 1) / threadGroupHeight,
                    depth: sourceImage.texture.arrayLength)
                encoder.dispatchThreadgroups(gridShape, threadsPerThreadgroup: threadGroupShape)
                encoder.endEncoding()
            }
            
            let destinationImage = self.destinationImage(sourceImage: sourceImage, commandBuffer: commandBuffer)
            convolution.encode(commandBuffer: commandBuffer, sourceImage: intermediateImage, destinationImage: destinationImage)
            outputMemoId = commandBuffer.hash
            outputMemo = destinationImage
            return outputMemo!
        }
    }
    
    func destinationImage(sourceImage: MPSImage, commandBuffer: MTLCommandBuffer) -> MPSImage {
        let inHeight = Int(sourceImage.height)
        let inWidth = Int(sourceImage.width)
        
        let outHeight = self.stride * (inHeight - 1) + self.kernelSize - 2 * self.padding
        let outWidth = self.stride * (inWidth - 1) + self.kernelSize - 2 * self.padding
        
        /* Assert the constraint on input size, kernel size, padding, stride. */
        assert((outHeight + 2 * padding - kernelSize) % self.stride == 0,
               "Input size must be a multiple of i+2p-k in all dimensions. This constraint is failing in the height dimension.")
        assert((outWidth + 2 * padding - kernelSize) % self.stride == 0,
               "Input size must be a multiple of i+2p-k in all dimensions. This constraint is failing in the width dimension.")
        
        let textureDesc = MTLTextureDescriptor()
        textureDesc.arrayLength = (channelsOut + 3) / 4
        textureDesc.height = outHeight
        textureDesc.width = outWidth
        textureDesc.textureType = .type2DArray
        textureDesc.usage = MTLTextureUsage(
            rawValue: MTLTextureUsage.shaderWrite.rawValue | MTLTextureUsage.shaderRead.rawValue)
        textureDesc.pixelFormat = .rgba16Float
        
        if useTemporary {
            let img = MPSTemporaryImage.init(
                commandBuffer: commandBuffer,
                textureDescriptor: textureDesc)
            img.readCount = consumerCount
            return img
        } else {
            let texture = ShaderRegistry.getDevice().makeTexture(descriptor: textureDesc)
            return MPSImage.init(texture: texture, featureChannels: max(4, channelsOut))
        }
    }
    
    /*  Transform weights so we can use convolution to calculate deconvolution.
        The transform is a vertical flip of the kernel height axis,
        then a horizontal flip of the kernel width axis. */
    static func transformWeights(
        weights w: ParameterBuffer,
        channelsOut co: Int,
        kernelSize nk: Int,
        channelsIn nci: Int) -> ParameterBuffer {
        
        let outCount = co * nk * nk * nci
        
        /* 4D array with dimensions (c_o, h, w, c_i) */
        let in_w = UnsafeBufferPointer<Float>(start: w.pointer, count: w.count)
        let out_w = UnsafeMutablePointer<Float>.allocate(capacity: outCount)
        for co in 0 ..< co {
            let co_off = co * (nk * nk * nci)
            for kh in 0 ..< nk {
                let kh_in_off = kh * (nk * nci)
                let kh_out_off = (nk - 1 - kh) * (nk * nci)
                for kw in 0 ..< nk {
                    let kw_in_off = kw * nci
                    let kw_out_off = (nk - 1 - kw) * nci
                    for ci in 0 ..< nci {
                        out_w[co_off + kh_out_off + kw_out_off + ci] =
                            in_w[co_off + kh_in_off + kw_in_off + ci]
                    }
                }
            }
        }
        
        return MemoryParameterBuffer(
            buffer: UnsafeBufferPointer<Float>.init(start: out_w, count: outCount))
    }
}

