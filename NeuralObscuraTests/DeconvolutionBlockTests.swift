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

class DeconvolutionBlockTests: CommandEncoderBaseTest {
    
    func testGroundTruthTensorDot() {
        let testUrl = Bundle(for: type(of: self))
            .url(forResource: "deconv_input", withExtension: "npy", subdirectory: "testdata")!
        let testImg = MPSImage.loadFromNumpy(testUrl)
        
        let w_pb = FileParameterBuffer(modelName: "composition", rawFileName: "d1_W")
        
        let tensorDot = TensorDotLayer(
            kernelSize: 4,
            channelsIn: 128,
            channelsOut: 64,
            w: w_pb)
        
        let outputBuf = tensorDot.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
        execute()
        
        let expUrl = Bundle(for: type(of: self))
            .url(forResource: "tensordot_expected_output", withExtension: "dat", subdirectory: "testdata")!
        let expBuf = MTLBufferUtil.loadFromBinary(expUrl)

        XCTAssert(MTLBufferUtil.lossyEqual(lhs: outputBuf, rhs: expBuf, precision: 0, type: UInt16.self))
    }
    
//    func testGroundTruthDeconv() {
//        let testUrl = Bundle(for: type(of: self))
//            .url(forResource: "deconv_input", withExtension: "npy", subdirectory: "testdata")!
//        let testImg = MPSImage.loadFromNumpy(testUrl, destinationPixelFormat: testTextureFormatRGBA)
//        
//        let w_pb = FileParameterBuffer(modelName: "composition", rawFileName: "d1_W")
//        let b_pb = FileParameterBuffer(modelName: "composition", rawFileName: "d1_b")
//        
//        let deconv = DeconvolutionBlock(
//            kernelSize: 4,
//            channelsIn: 128,
//            channelsOut: 64,
//            w: w_pb,
//            b: b_pb,
//            relu: false,
//            padding: 1,
//            stride: 2)
//        
//        let outputImg = deconv.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
//        execute()
//        
//        let expUrl = Bundle(for: type(of: self))
//            .url(forResource: "deconv_expected_output", withExtension: "npy", subdirectory: "testdata")!
//        let expImg = MPSImage.loadFromNumpy(expUrl, destinationPixelFormat: testTextureFormatRGBA)
//        
//        XCTAssertEqual(outputImg, expImg)
//    }
    
    func testGroundTruthCol2Im() {
        let inputImageUrl = Bundle(for: type(of: self))
            .url(forResource: "tensordot_input", withExtension: "npy", subdirectory: "testdata")!
        let inputImage = MPSImage.loadFromNumpy(inputImageUrl)
        
        let inputBufferUrl = Bundle(for: type(of: self))
            .url(forResource: "col2im_input", withExtension: "dat", subdirectory: "testdata")!
        let inputBuffer = MTLBufferUtil.loadFromBinary(inputBufferUrl)
        
        let expectedOutputUrl = Bundle(for: type(of: self))
            .url(forResource: "col2im_expected_output", withExtension: "npy", subdirectory: "testdata")!
        let expectedOutput = MPSImage.loadFromNumpy(expectedOutputUrl)
        
        let col2Im = Col2ImLayer(channelsOut: 64, kernelSize: 4, stride: 2, padding: 1)
        let output = col2Im.chain(MTLBufferVariable(inputBuffer), MPSImageVariable(inputImage)).forward(commandBuffer: commandBuffer)
        execute()
        
        XCTAssert(output.isLossyEqual(image: expectedOutput, precision: 2))
    }
    
    //         func testIdentityNoPadding() {
    //             let testImg = device.MakeMPSImage(width: 2, height: 2, values: [1, 0,
    //                                                                             0, 1] as [Float32])
    //    
    //             /* Create our CommandEncoder */
    //             let w_pb = MemoryParameterBuffer([0, 0, 0,
    //                                               0, 1, 0,
    //                                               0, 0, 0])
    //             let b_pb = MemoryParameterBuffer(0)
    //             let deconv = DeconvolutionBlock(
    //                 kernelSize: 3,
    //                 channelsIn: 1,
    //                 channelsOut: 1,
    //                 w: w_pb,
    //                 b: b_pb,
    //                 relu: false,
    //                 padding: 0)
    //    
    //             let expImg = device.MakeMPSImage(width: 4,
    //                                              height: 4,
    //                                              pixelFormat: testTextureFormatR,
    //                                              values: [0, 0, 0, 0,
    //                                                       0, 1, 0, 0,
    //                                                       0, 0, 1, 0,
    //                                                       0, 0, 0, 0] as [Float32])
    //    
    //             /* Run our test */
    //             let outputImg = deconv.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
    //             execute()
    //    
    //             /* Verify the result */
    //             XCTAssertEqual(outputImg, expImg)
    //         }
    //    
    //         func testIdentityHalfPadding() {
    //             let testImg = device.MakeMPSImage(width: 4,
    //                                               height: 4,
    //                                               values: [0, 0, 0, 0,
    //                                                        0, 1, 0, 0,
    //                                                        0, 0, 1, 0,
    //                                                        0, 0, 0, 0] as [Float32])
    //    
    //             /* Create our CommandEncoder */
    //             let w_pb = MemoryParameterBuffer([0, 0, 0,
    //                                               0, 1, 0,
    //                                               0, 0, 0])
    //             let b_pb = MemoryParameterBuffer(0)
    //             let deconv = DeconvolutionBlock(
    //                 kernelSize: 3,
    //                 channelsIn: 1,
    //                 channelsOut: 1,
    //                 w: w_pb,
    //                 b: b_pb,
    //                 relu: false,
    //                 padding: 1)
    //    
    //             let expImg = device.MakeMPSImage(width: 4,
    //                                              height: 4,
    //                                              pixelFormat: testTextureFormatR,
    //                                              values: [0, 0, 0, 0,
    //                                                       0, 1, 0, 0,
    //                                                       0, 0, 1, 0,
    //                                                       0, 0, 0, 0] as [Float32])
    //    
    //             /* Run our test */
    //             let outputImg = deconv.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
    //             execute()
    //    
    //             /* Verify the result */
    //             XCTAssertEqual(outputImg, expImg)
    //         }
    //    
    //         func testIdentityFullPadding() {
    //             let testImg = device.MakeMPSImage(width: 6,
    //                                               height: 6,
    //                                               values: [0, 0, 0, 0, 0, 0,
    //                                                        0, 0, 0, 0, 0, 0,
    //                                                        0, 0, 1, 0, 1, 0,
    //                                                        0, 0, 0, 1, 0, 0,
    //                                                        0, 0, 0, 0, 0, 0,
    //                                                        0, 0, 0, 0, 0, 0] as [Float32])
    //             /* Create our CommandEncoder */
    //             let w_pb = MemoryParameterBuffer([0, 0, 0,
    //                                               0, 1, 0,
    //                                               0, 0, 0])
    //             let b_pb = MemoryParameterBuffer(0)
    //             let deconv = DeconvolutionBlock(
    //                 kernelSize: 3,
    //                 channelsIn: 1,
    //                 channelsOut: 1,
    //                 w: w_pb,
    //                 b: b_pb,
    //                 relu: false,
    //                 padding: 2)
    //    
    //             let expImg = device.MakeMPSImage(width: 4,
    //                                              height: 4,
    //                                              pixelFormat: testTextureFormatR,
    //                                              values: [0, 0, 0, 0,
    //                                                       0, 1, 0, 1,
    //                                                       0, 0, 1, 0,
    //                                                       0, 0, 0, 0] as [Float32])
    //    
    //             /* Run our test */
    //             let outputImg = deconv.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
    //             execute()
    //    
    //             /* Verify the result */
    //             XCTAssertEqual(outputImg, expImg)
    //         }
    //    
    //         func testSumHalfPaddingNonUnitStride() {
    //             let testImg = device.MakeMPSImage(width: 4,
    //                                               height: 4,
    //                                               values: [0, 0, 0, 0,
    //                                                        0, 1, 0, 0,
    //                                                        0, 0, 1, 0,
    //                                                        0, 0, 0, 0] as [Float32])
    //    
    //             /* Create our CommandEncoder */
    //             let w_pb = MemoryParameterBuffer([1, 1, 1,
    //                                               1, 1, 1,
    //                                               1, 1, 1])
    //             let b_pb = MemoryParameterBuffer(0)
    //             let deconv = DeconvolutionBlock(
    //                 kernelSize: 3,
    //                 channelsIn: 1,
    //                 channelsOut: 1,
    //                 w: w_pb,
    //                 b: b_pb,
    //                 relu: false,
    //                 padding: 1,
    //                 stride: 2)
    //    
    //             let expImg = device.MakeMPSImage(width: 7,
    //                                              height: 7,
    //                                              pixelFormat: testTextureFormatR,
    //                                              values: [0, 0, 0, 0, 0, 0, 0,
    //                                                       0, 1, 1, 1, 0, 0, 0,
    //                                                       0, 1, 1, 1, 0, 0, 0,
    //                                                       0, 1, 1, 2, 1, 1, 0,
    //                                                       0, 0, 0, 1, 1, 1, 0,
    //                                                       0, 0, 0, 1, 1, 1, 0,
    //                                                       0, 0, 0, 0, 0, 0, 0] as [Float32])
    //             
    //             /* Run our test */
    //             let outputImg = deconv.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
    //             execute()
    //             
    //             /* Verify the result */
    //             XCTAssertEqual(outputImg, expImg)
    //         }
}
