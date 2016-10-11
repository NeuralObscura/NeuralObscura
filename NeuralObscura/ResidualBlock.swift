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
    let c1, c2: BatchNormalizationLayer
    
    
    /* A property to keep info from init time whether we will pad input image or not for use during encode call */
    fileprivate var padding = true
    
    init(
        device: MTLDevice,
        modelParams: [String: StyleModelData],
        blockName: String,
        channelsIn: UInt,
        channelsOut: UInt,
        kernelSize: UInt = 3,
        stride: Int = 1,
        useTemporary: Bool = true) {
        
        /* Load the block parameters */
        let c1_w = modelParams[blockName + "_c1_W"]!
        let c1_b = modelParams[blockName + "_c1_b"]!
        let c2_w = modelParams[blockName + "_c2_W"]!
        let c2_b = modelParams[blockName + "_c2_b"]!
        
        /* Init block encoders */
        // TODO: Disable relu!
        c1 = BatchNormalizationLayer(
            device: device,
            kernelSize: kernelSize,
            channelsIn: channelsIn,
            channelsOut: channelsOut,
            w: c1_w,
            b: c1_b,
            relu: false,
            padding: 1,
            stride: stride,
            useTemporary: useTemporary)
        
        // TODO: Disable relu!
        c2 = BatchNormalizationLayer(
            device: device,
            kernelSize: kernelSize,
            channelsIn: channelsOut,
            channelsOut: channelsOut,
            w: c2_w,
            b: c2_b,
            relu: false,
            padding: 1,
            stride: stride,
            useTemporary: useTemporary)

    }
    
    func chain(_ top: CommandEncoder) -> CommandEncoder {
        var h: CommandEncoder
        
        // TODO: Add relu around b1
        h = c1.chain(top)
        
        h = c2.chain(h)
        
        // TODO: Set up summation layer
        
        return h
    }
    
}
