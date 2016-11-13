//
//  ResidualBlock.swift
//  NeuralObscura
//
//  Created by Edward Knox on 9/25/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders


/**
 * Blocks simulate a layer by encapsulating multiple inner layers, structured like a sub-network.
 * chain() and encode() methods should not expose any direct linkages to inner layers
 */
class ResidualBlock: UnaryChain {
    let c1, c2: ConvolutionLayer
    let b1, b2: BatchNormalizationLayer
    let s1: ResidualBlockSummationLayer
    
    
    /* A property to keep info from init time whether we will pad input image or not for use during encode call */
    fileprivate var padding = true
    
    init(
        modelParams: [String: ParameterBuffer],
        blockName: String,
        channelsIn: Int,
        channelsOut: Int,
        kernelSize: Int = 3,
        stride: Int = 1,
        debug: Bool = false) {
        
        /* Load the block parameters */
        let c1_w = modelParams[blockName + "_c1_W"]!
        let c1_b = modelParams[blockName + "_c1_b"]!
        let c2_w = modelParams[blockName + "_c2_W"]!
        let c2_b = modelParams[blockName + "_c2_b"]!
        let b1_beta = modelParams[blockName + "_b1_beta"]!
        let b1_gamma = modelParams[blockName + "_b1_gamma"]!
        let b2_beta = modelParams[blockName + "_b2_beta"]!
        let b2_gamma = modelParams[blockName + "_b2_gamma"]!
        
        /* Init block encoders */
        c1 = ConvolutionLayer(
            kernelSize: kernelSize,
            channelsIn: channelsIn,
            channelsOut: channelsOut,
            w: c1_w,
            b: c1_b,
            relu: false,
            padding: 1,
            stride: stride,
            debug: debug)

        c2 = ConvolutionLayer(
            kernelSize: kernelSize,
            channelsIn: channelsOut,
            channelsOut: channelsOut,
            w: c2_w,
            b: c2_b,
            relu: false,
            padding: 1,
            stride: stride,
            debug: debug)

        b1 = BatchNormalizationLayer(channelsIn: channelsOut,
                                     beta: b1_beta,
                                     gamma: b1_gamma)
        b2 = BatchNormalizationLayer(channelsIn: channelsOut,
                                     beta: b2_beta,
                                     gamma: b2_gamma)
        s1 = ResidualBlockSummationLayer()
        
    }
    
    func chain(_ top: CommandEncoder) -> CommandEncoder {
        // TODO: Add relu around b1
        var h = b1.chain(top)
        
        h = b2.chain(c2.chain(h))
        
        return s1.chain(top, h)
    }
    
}

class ResidualBlockSummationLayer: BinaryCommandEncoder {
    init(debug: Bool = false) {
        super.init(
            delegate: ResidualBlockSummationLayerDelegate(),
            debug: debug)
    }
}

class ResidualBlockSummationLayerDelegate: CommandEncoderDelegate {
    private var sourceImageA: MPSImage!
    private var sourceImageB: MPSImage!
    private var inputsSupplied = 0
    
    init() {}

    func getDestinationImageDescriptor(sourceImage: MPSImage) -> MPSImageDescriptor {
        return MPSImageDescriptor(
            channelFormat: textureFormat,
            width: sourceImage.width,
            height: sourceImage.height,
            featureChannels: sourceImage.featureChannels)
    }
    
    
    func supplyInput(sourceImage: MPSImage, sourcePosition: Int) -> Bool {
        switch sourcePosition {
        case 0:
            self.sourceImageA = sourceImage
        case 1:
            self.sourceImageB = sourceImage
        default:
            fatalError("Invalid sourcePosition: \(sourcePosition)")
        }
        inputsSupplied += 1
        return inputsSupplied == 2
    }

    func encode(commandBuffer: MTLCommandBuffer, destinationImage: MPSImage) {
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
    }
}
