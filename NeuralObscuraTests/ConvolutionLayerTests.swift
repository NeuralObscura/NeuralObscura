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
        // Build test model params
        let w = MemoryParameterBuffer([0, 0, 0,
                                       0, 1, 0,
                                       0, 0, 0])
        let b = MemoryParameterBuffer(1)  
        
        let conv = ConvolutionLayer(
            device: device,
            kernelSize: 3,
            channelsIn: 1,
            channelsOut: 1,
            w: w,
            b: b,
            relu: false,
            debug: false)
        
        // Create a test image
//        let testImage = MPSImage(device: device,
//                               width: 4,
//                               values: [1, 1, 1, 1,
//                                        1, 1, 1, 1,
//                                        1, 1, 1, 1,
//                                        1, 1, 1, 1])
        let testImage = MPSImage(device: device,
                               width: 3,
                               values: [1, 1, 1,
                                        1, 1, 1,
                                        1, 1, 1])
        print(testImage)
        // Execute test subject
        let testOutput = conv.execute(commandBuffer: commandBuffer, sourceImage: testImage)
        
        // print("test output width: \(testOutput.width)")
        // print("test output height: \(testOutput.height)")
        // print("test output buffer length: \(testOutput.texture.buffer!.length)")
        // print("test output:")
         print(testOutput)
        // Create expected output image
        // let expectedOutput: MPSImage
        // 
        // assertEqual(testOutput, expectedOutput)
    }
    
}
