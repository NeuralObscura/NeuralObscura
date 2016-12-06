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
    let s1: SummationLayer
    let r1: ReLULayer
    
    
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
                                     gamma: b1_gamma,
                                     mean: b1_mean,
                                     stddev: b1_stddev)
        b2 = BatchNormalizationLayer(channelsIn: channelsOut,
                                     beta: b2_beta,
                                     gamma: b2_gamma,
                                     mean: b2_mean,
                                     stddev: b2_stddev)
        s1 = SummationLayer()
        
        r1 = ReLULayer()
        
    }
    
    func chain(_ top: CommandEncoder) -> CommandEncoder {

        var h = r1.chain(b1.chain(c1.chain(top)))
        
        h = b2.chain(c2.chain(h))
        
        return s1.chain(top, h)
    }
    
}
