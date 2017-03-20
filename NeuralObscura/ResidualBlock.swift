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
class ResidualBlock: UnaryCommandEncoder {
    private let c1, c2: ConvolutionLayer
    private let b1, b2: BatchNormalizationLayer
    private let r1: ReLULayer
    private let s1: SummationLayer
    
    /* A property to keep info from init time whether we will pad input image or not for use during encode call */
    fileprivate var padding = true
    
    init(
        modelParams: [String: ParameterBuffer],
        blockName: String,
        channelsIn: Int,
        channelsOut: Int,
        kernelSize: Int = 3,
        stride: Int = 1,
        useTemporary: Bool = false) {
        
        /* Load the block parameters */
        let c1_w = modelParams[blockName + "_c1_W"]!
        let c1_b = modelParams[blockName + "_c1_b"]!
        let c2_w = modelParams[blockName + "_c2_W"]!
        let c2_b = modelParams[blockName + "_c2_b"]!
        let b1_beta = modelParams[blockName + "_b1_beta"]!
        let b1_gamma = modelParams[blockName + "_b1_gamma"]!
        let b1_mean = modelParams[blockName + "_b1_mean"]!
        let b1_stddev = modelParams[blockName + "_b1_stddev"]!
        let b2_beta = modelParams[blockName + "_b2_beta"]!
        let b2_gamma = modelParams[blockName + "_b2_gamma"]!
        let b2_mean = modelParams[blockName + "_b2_mean"]!
        let b2_stddev = modelParams[blockName + "_b2_stddev"]!

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
            useTemporary: useTemporary)

        b1 = BatchNormalizationLayer(channelsIn: channelsOut,
                                     beta: b1_beta,
                                     gamma: b1_gamma,
                                     mean: b1_mean,
                                     stddev: b1_stddev)

        r1 = ReLULayer()
        
        c2 = ConvolutionLayer(
            kernelSize: kernelSize,
            channelsIn: channelsOut,
            channelsOut: channelsOut,
            w: c2_w,
            b: c2_b,
            relu: false,
            padding: 1,
            stride: stride,
            useTemporary: useTemporary)
        
        b2 = BatchNormalizationLayer(channelsIn: channelsOut,
                                     beta: b2_beta,
                                     gamma: b2_gamma,
                                     mean: b2_mean,
                                     stddev: b2_stddev)
        
        s1 = SummationLayer()
    }
    
    func chain(_ top: AnyCommandEncoder<MPSImage>) -> AnyCommandEncoder<MPSImage> {
        var h = r1.chain(b1.chain(c1.chain(top)))
        h = b2.chain(c2.chain(h))
        h = s1.chain(top, h)
        return h
    }
    
    func registerConsumer() {
        s1.registerConsumer()
    }
    
    func forward(commandBuffer: MTLCommandBuffer) -> MPSImage {
        return s1.forward(commandBuffer: commandBuffer)
    }
}
