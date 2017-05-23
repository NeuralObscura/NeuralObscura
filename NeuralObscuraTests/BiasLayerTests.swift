//
//  BiasLayerTests.swift
//  NeuralObscura
//
//  Created by Edward Knox on 5/8/17.
//  Copyright © 2017 Paul Bergeron. All rights reserved.
//

import Foundation
import XCTest
import MetalKit
import MetalPerformanceShaders
@testable import NeuralObscura

class BiasLayerTests: CommandEncoderBaseTest {
    
    func testBiasGroundTruth() throws {
        let testUrl = Bundle(for: type(of: self))
            .url(forResource: "deconv_bias_input", withExtension: "npy", subdirectory: "testdata")!
        let testImg = MPSImage.fromNumpy(testUrl)
        
        let b_pb = FileParameterBuffer(modelName: "composition", rawFileName: "d1_b")
        
        let bias = BiasLayer(biases: b_pb, useTemporary: false)
        
        /* Run our test */
        let outputImg = bias.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
        execute()
        
        let expUrl = Bundle(for: type(of: self))
            .url(forResource: "deconv_bias_expected_output", withExtension: "npy", subdirectory: "testdata")!
        let expImg = MPSImage.fromNumpy(expUrl)
        
        XCTAssert(outputImg.isLossyEqual(image: expImg, precision: 1))
    }
}

