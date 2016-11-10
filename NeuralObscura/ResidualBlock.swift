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
class ResidualBlock: Chain {
    let c1, c2: ConvolutionLayer
    let b1, b2: BatchNormalizationLayer
    
    
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
        // TODO: Disable relu!
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
        
        // TODO: Disable relu!
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
        
    }
    
    func chain(_ top: CommandEncoder) -> CommandEncoder {
        var h: CommandEncoder
        
        // TODO: Add relu around b1
        h = b1.chain(top)
        
        h = b2.chain(c2.chain(h))
        
        // TODO: Set up summation layer
        
        return h
    }
    
}
