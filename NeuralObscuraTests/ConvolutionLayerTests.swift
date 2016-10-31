//
//  CommandEncoderBaseTest.swift
//  NeuralObscura
//
//  Created by Edward Knox on 10/9/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import XCTest
import MetalKit
import MetalPerformanceShaders
@testable import NeuralObscura

class ConvolutionLayerTests: CommandEncoderBaseTest {
    
    func testConvolution() {
        let testImg = device.MakeTestMPSImage(width: 4, height: 4, values: [0, 0, 0, 0,
                                                                            0, 1, 0, 1,
                                                                            0, 0, 1, 0,
                                                                            0, 0, 0, 0])
        
        /* Create our CommandEncoder */
        let w: [Float] = [0, 0, 0,
                          0, 1, 0,
                          0, 0, 0]
        var b: Float = 0
        let w_pb = MemoryParameterBuffer(w)
        let b_pb = MemoryParameterBuffer(b)
        let conv = ConvolutionLayer(
            device: device,
            kernelSize: 3,
            channelsIn: 1,
            channelsOut: 1,
            w: w_pb,
            b: b_pb,
            relu: false,
            padding: 1,
            debug: true)
        
        let expImg = device.MakeTestMPSImage(width: 4, height: 4, values: [0, 0, 0, 0,
                                                                           0, 1, 0, 1,
                                                                           0, 0, 1, 0,
                                                                           0, 0, 0, 0])
        
        
        /* Run our test */
        let outputImg = conv.execute(commandBuffer: commandBuffer, sourceImage: testImg)
        
        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }
    
}
