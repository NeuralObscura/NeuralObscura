//
//  DeconvolutionLayerTests.swift
//  NeuralObscura
//
//  Created by Edward Knox on 11/4/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import XCTest
import MetalKit
import MetalPerformanceShaders
@testable import NeuralObscura

class DeconvolutionLayerTests: CommandEncoderBaseTest {
    
    func testIdentityNoPadding() {
        let testImg = device.MakeTestMPSImage(width: 2, height: 2, values: [1, 0,
                                                                            0, 1])
        
        /* Create our CommandEncoder */
        let w_pb = MemoryParameterBuffer([0, 0, 0,
                                          0, 1, 0,
                                          0, 0, 0])
        let b_pb = MemoryParameterBuffer(0)
        let deconv = DeconvolutionLayer(
            kernelSize: 3,
            channelsIn: 1,
            channelsOut: 1,
            w: w_pb,
            b: b_pb,
            relu: false,
            padding: 0,
            outputType: CommandEncoderOutputType.debug)
        
        let expImg = device.MakeTestMPSImage(width: 4, height: 4, values: [0, 0, 0, 0,
                                                                           0, 1, 0, 0,
                                                                           0, 0, 1, 0,
                                                                           0, 0, 0, 0])
        
        /* Run our test */
        let outputImg = deconv.execute(commandBuffer: commandBuffer, sourceImage: testImg)
        
        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }
    
    func testIdentityHalfPadding() {
        let testImg = device.MakeTestMPSImage(width: 4, height: 4, values: [0, 0, 0, 0,
                                                                            0, 1, 0, 0,
                                                                            0, 0, 1, 0,
                                                                            0, 0, 0, 0])
        
        /* Create our CommandEncoder */
        let w_pb = MemoryParameterBuffer([0, 0, 0,
                                          0, 1, 0,
                                          0, 0, 0])
        let b_pb = MemoryParameterBuffer(0)
        let deconv = DeconvolutionLayer(
            kernelSize: 3,
            channelsIn: 1,
            channelsOut: 1,
            w: w_pb,
            b: b_pb,
            relu: false,
            padding: 1,
            outputType: CommandEncoderOutputType.debug)
        
        let expImg = device.MakeTestMPSImage(width: 4, height: 4, values: [0, 0, 0, 0,
                                                                           0, 1, 0, 0,
                                                                           0, 0, 1, 0,
                                                                           0, 0, 0, 0])
        
        /* Run our test */
        let outputImg = deconv.execute(commandBuffer: commandBuffer, sourceImage: testImg)
        
        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }
    
    func testIdentityFullPadding() {
        let testImg = device.MakeTestMPSImage(width: 6, height: 6, values: [0, 0, 0, 0, 0, 0,
                                                                            0, 0, 0, 0, 0, 0,
                                                                            0, 0, 1, 0, 1, 0,
                                                                            0, 0, 0, 1, 0, 0,
                                                                            0, 0, 0, 0, 0, 0,
                                                                            0, 0, 0, 0, 0, 0])
        /* Create our CommandEncoder */
        let w_pb = MemoryParameterBuffer([0, 0, 0,
                                          0, 1, 0,
                                          0, 0, 0])
        let b_pb = MemoryParameterBuffer(0)
        let deconv = DeconvolutionLayer(
            kernelSize: 3,
            channelsIn: 1,
            channelsOut: 1,
            w: w_pb,
            b: b_pb,
            relu: false,
            padding: 2,
            outputType: CommandEncoderOutputType.debug)
        
        let expImg = device.MakeTestMPSImage(width: 4, height: 4, values: [0, 0, 0, 0,
                                                                           0, 1, 0, 1,
                                                                           0, 0, 1, 0,
                                                                           0, 0, 0, 0])
        
        /* Run our test */
        let outputImg = deconv.execute(commandBuffer: commandBuffer, sourceImage: testImg)
        
        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }
    
    func testSumHalfPaddingNonUnitStride() {
        let testImg = device.MakeTestMPSImage(width: 4, height: 4, values: [0, 0, 0, 0,
                                                                            0, 1, 0, 0,
                                                                            0, 0, 1, 0,
                                                                            0, 0, 0, 0])
        
        /* Create our CommandEncoder */
        let w_pb = MemoryParameterBuffer([1, 1, 1,
                                          1, 1, 1,
                                          1, 1, 1])
        let b_pb = MemoryParameterBuffer(0)
        let deconv = DeconvolutionLayer(
            kernelSize: 3,
            channelsIn: 1,
            channelsOut: 1,
            w: w_pb,
            b: b_pb,
            relu: false,
            padding: 1,
            stride: 2,
            outputType: CommandEncoderOutputType.debug)
        
        let expImg = device.MakeTestMPSImage(width: 7, height: 7, values: [0, 0, 0, 0, 0, 0, 0,
                                                                           0, 1, 1, 1, 0, 0, 0,
                                                                           0, 1, 1, 1, 0, 0, 0,
                                                                           0, 1, 1, 2, 1, 1, 0,
                                                                           0, 0, 0, 1, 1, 1, 0,
                                                                           0, 0, 0, 1, 1, 1, 0,
                                                                           0, 0, 0, 0, 0, 0, 0])
        
        /* Run our test */
        let outputImg = deconv.execute(commandBuffer: commandBuffer, sourceImage: testImg)
        
        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }
    
}
