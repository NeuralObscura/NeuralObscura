//
//  DeconvolutionLayerV2Tests.swift
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

class DeconvolutionLayerV2Tests: CommandEncoderBaseTest {
    
    func testGroundTruth1() {
        let testUrl = Bundle(for: type(of: self))
            .url(forResource: "deconv_input", withExtension: "npy", subdirectory: "testdata")!
        let testImg = MPSImage.fromNumpy(testUrl)
        
        let w_pb = FileParameterBuffer(modelName: "composition", rawFileName: "d1_W")
        let b_pb = FileParameterBuffer(modelName: "composition", rawFileName: "d1_b")
        
        let deconv = DeconvolutionLayerV2(
            kernelSize: 4,
            channelsIn: 128,
            channelsOut: 64,
            w: w_pb,
            b: b_pb,
            relu: false,
            padding: 1,
            stride: 2)
        
        let outputImg = deconv.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
        execute()
        
        let expUrl = Bundle(for: type(of: self))
            .url(forResource: "deconv_expected_output", withExtension: "npy", subdirectory: "testdata")!
        let expImg = MPSImage.fromNumpy(expUrl)
        
        XCTAssert(outputImg.isLossyEqual(image: expImg, precision: 0))
    }
    
    func testGroundTruth2() {
        let testUrl = Bundle(for: type(of: self))
            .url(forResource: "deconv_input_2", withExtension: "npy", subdirectory: "testdata")!
        let testImg = MPSImage.fromNumpy(testUrl)
        
        let w_pb = FileParameterBuffer(modelName: "composition", rawFileName: "d3_W")
        let b_pb = FileParameterBuffer(modelName: "composition", rawFileName: "d3_b")
        
        let deconv = DeconvolutionLayerV2(
            kernelSize: 9,
            channelsIn: 32,
            channelsOut: 3,
            w: w_pb,
            b: b_pb,
            relu: false,
            padding: 4,
            stride: 1)
        
        let outputImg = deconv.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
        execute()
        
        let expUrl = Bundle(for: type(of: self))
            .url(forResource: "deconv_expected_output_2", withExtension: "npy", subdirectory: "testdata")!
        let expImg = MPSImage.fromNumpy(expUrl)
        
        XCTAssert(outputImg.isLossyEqual(image: expImg, precision: 0))
    }
    
    func testLargeDeconv() {
        let testUrl = Bundle(for: type(of: self))
            .url(forResource: "large_deconv_input", withExtension: "npy", subdirectory: "testdata")!
        let testImg = MPSImage.fromNumpy(testUrl)
        
        let w_pb = FileParameterBuffer(modelName: "composition", rawFileName: "d1_W")
        let b_pb = FileParameterBuffer(modelName: "composition", rawFileName: "d1_b")
        
        let deconv = DeconvolutionLayerV2(
            kernelSize: 4,
            channelsIn: 128,
            channelsOut: 64,
            w: w_pb,
            b: b_pb,
            relu: false,
            padding: 1,
            stride: 2)
        
        _ = deconv.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
        execute()
        
        if ((commandBuffer.error) != nil) {
            XCTAssert(false, commandBuffer.error!.localizedDescription)
        } else {
            XCTAssert(true)
        }
    }
}
