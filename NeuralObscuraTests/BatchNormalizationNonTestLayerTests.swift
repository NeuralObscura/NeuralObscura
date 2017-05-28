//
//  BatchNormalizationNonTestLayerTests.swift
//  NeuralObscura
//
//  Created by Edward Knox on 5/8/17.
//  Copyright © 2017 Paul Bergeron. All rights reserved.
//

import XCTest
import MetalKit
import MetalPerformanceShaders
@testable import NeuralObscura

class BatchNormalizationNonTestLayerTests: CommandEncoderBaseTest {

    func testGroundTruthBatchNormNonTest() {
        let testUrl = Bundle(for: type(of: self))
            .url(forResource: "batch_norm_input", withExtension: "npy", subdirectory: "testdata")!
        let testImg = MPSImage.fromNumpy(testUrl)
        
        let expUrl = Bundle(for: type(of: self))
            .url(forResource: "batch_norm_expected_output", withExtension: "npy", subdirectory: "testdata")!
        let expImg = MPSImage.fromNumpy(expUrl)

        /* Create our CommandEncoder*/
        let gamma = FileParameterBuffer(modelName: "composition", rawFileName: "b1_gamma")
        let beta = FileParameterBuffer(modelName: "composition", rawFileName: "b1_beta")
        
        let bn =  BatchNormalizationNonTestLayer(beta: beta, gamma: gamma)
        
        /* Run our test */
        let outputImg = bn.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
        execute()
        
        XCTAssert(outputImg.isLossyEqual(image: expImg, precision: 0))
    }
}
