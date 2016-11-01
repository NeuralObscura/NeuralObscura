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
    
    func testIdentityNoPadding() {
        let testImg = device.MakeTestMPSImage(width: 4, height: 4, values: [0, 0, 0, 0,
                                                                            0, 1, 0, 1,
                                                                            0, 0, 1, 0,
                                                                            0, 0, 0, 0])
        
        /* Create our CommandEncoder */
        let w_pb = MemoryParameterBuffer([0, 0, 0,
                                          0, 1, 0,
                                          0, 0, 0])
        let b_pb = MemoryParameterBuffer(0)
        let conv = ConvolutionLayer(
            device: device,
            kernelSize: 3,
            channelsIn: 1,
            channelsOut: 1,
            w: w_pb,
            b: b_pb,
            relu: false,
            padding: 0,
            outputType: NeuralStyleModelLayerOutputType.debug)
        
        let expImg = device.MakeTestMPSImage(width: 2, height: 2, values: [1, 0,
                                                                           0, 1])
        
        /* Run our test */
        let outputImg = conv.execute(commandBuffer: commandBuffer, sourceImage: testImg)
        
        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }
    
    func testIdentityHalfPadding() {
        let testImg = device.MakeTestMPSImage(width: 4, height: 4, values: [0, 0, 0, 0,
                                                                            0, 1, 0, 1,
                                                                            0, 0, 1, 0,
                                                                            0, 0, 0, 0])
        
        /* Create our CommandEncoder */
        let w_pb = MemoryParameterBuffer([0, 0, 0,
                                          0, 1, 0,
                                          0, 0, 0])
        let b_pb = MemoryParameterBuffer(0)
        let conv = ConvolutionLayer(
            device: device,
            kernelSize: 3,
            channelsIn: 1,
            channelsOut: 1,
            w: w_pb,
            b: b_pb,
            relu: false,
            padding: 1,
            outputType: NeuralStyleModelLayerOutputType.debug)
        
        let expImg = device.MakeTestMPSImage(width: 4, height: 4, values: [0, 0, 0, 0,
                                                                           0, 1, 0, 1,
                                                                           0, 0, 1, 0,
                                                                           0, 0, 0, 0])
        
        
        /* Run our test */
        let outputImg = conv.execute(commandBuffer: commandBuffer, sourceImage: testImg)
        
        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }

    func testIdentityFullPadding() {
        let testImg = device.MakeTestMPSImage(width: 4, height: 4, values: [0, 0, 0, 0,
                                                                            0, 1, 0, 1,
                                                                            0, 0, 1, 0,
                                                                            0, 0, 0, 0])
        
        /* Create our CommandEncoder */
        let w_pb = MemoryParameterBuffer([0, 0, 0,
                                          0, 1, 0,
                                          0, 0, 0])
        let b_pb = MemoryParameterBuffer(0)
        let conv = ConvolutionLayer(
            device: device,
            kernelSize: 3,
            channelsIn: 1,
            channelsOut: 1,
            w: w_pb,
            b: b_pb,
            relu: false,
            padding: 2,
            outputType: NeuralStyleModelLayerOutputType.debug)
        
        let expImg = device.MakeTestMPSImage(width: 6, height: 6, values: [0, 0, 0, 0, 0, 0,
                                                                           0, 0, 0, 0, 0, 0,
                                                                           0, 0, 1, 0, 1, 0,
                                                                           0, 0, 0, 1, 0, 0,
                                                                           0, 0, 0, 0, 0, 0,
                                                                           0, 0, 0, 0, 0, 0])
        
        
        /* Run our test */
        let outputImg = conv.execute(commandBuffer: commandBuffer, sourceImage: testImg)
        
        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }
    
    func testSumNoPadding() {
        let testImg = device.MakeTestMPSImage(width: 4, height: 4, values: [0, 0, 0, 0,
                                                                            0, 1, 0, 1,
                                                                            0, 0, 1, 0,
                                                                            0, 0, 0, 0])
        
        /* Create our CommandEncoder */
        let w_pb = MemoryParameterBuffer([1, 1, 1,
                                          1, 1, 1,
                                          1, 1, 1])
        let b_pb = MemoryParameterBuffer(0)
        let conv = ConvolutionLayer(
            device: device,
            kernelSize: 3,
            channelsIn: 1,
            channelsOut: 1,
            w: w_pb,
            b: b_pb,
            relu: false,
            padding: 0,
            outputType: NeuralStyleModelLayerOutputType.debug)
        
        let expImg = device.MakeTestMPSImage(width: 2, height: 2, values: [2, 3,
                                                                           2, 3])
        
        /* Run our test */
        let outputImg = conv.execute(commandBuffer: commandBuffer, sourceImage: testImg)
        
        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }
    
    func testSumHalfPadding() {
        let testImg = device.MakeTestMPSImage(width: 4, height: 4, values: [0, 0, 0, 0,
                                                                            0, 1, 0, 1,
                                                                            0, 0, 1, 0,
                                                                            0, 0, 0, 0])
        
        /* Create our CommandEncoder */
        let w_pb = MemoryParameterBuffer([1, 1, 1,
                                          1, 1, 1,
                                          1, 1, 1])
        let b_pb = MemoryParameterBuffer(0)
        let conv = ConvolutionLayer(
            device: device,
            kernelSize: 3,
            channelsIn: 1,
            channelsOut: 1,
            w: w_pb,
            b: b_pb,
            relu: false,
            padding: 1,
            outputType: NeuralStyleModelLayerOutputType.debug)
        
        let expImg = device.MakeTestMPSImage(width: 4, height: 4, values: [1, 1, 2, 1,
                                                                           1, 2, 3, 2,
                                                                           1, 2, 3, 2,
                                                                           0, 1, 1, 1])
        
        
        /* Run our test */
        let outputImg = conv.execute(commandBuffer: commandBuffer, sourceImage: testImg)
        
        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }
    
    func testSumFullPadding() {
        let testImg = device.MakeTestMPSImage(width: 4, height: 4, values: [0, 0, 0, 0,
                                                                            0, 1, 0, 1,
                                                                            0, 0, 1, 0,
                                                                            0, 0, 0, 0])
        
        /* Create our CommandEncoder */
        let w_pb = MemoryParameterBuffer([1, 1, 1,
                                          1, 1, 1,
                                          1, 1, 1])
        let b_pb = MemoryParameterBuffer(0)
        let conv = ConvolutionLayer(
            device: device,
            kernelSize: 3,
            channelsIn: 1,
            channelsOut: 1,
            w: w_pb,
            b: b_pb,
            relu: false,
            padding: 2,
            outputType: NeuralStyleModelLayerOutputType.debug)
        
        let expImg = device.MakeTestMPSImage(width: 6, height: 6, values: [0, 0, 0, 0, 0, 0,
                                                                           0, 1, 1, 2, 1, 1,
                                                                           0, 1, 2, 3, 2, 1,
                                                                           0, 1, 2, 3, 2, 1,
                                                                           0, 0, 1, 1, 1, 0,
                                                                           0, 0, 0, 0, 0, 0])
        
        
        /* Run our test */
        let outputImg = conv.execute(commandBuffer: commandBuffer, sourceImage: testImg)
        
        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }
    
}
