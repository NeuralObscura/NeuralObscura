//
//  IncrementalModelGroundTruthTests.swift
//  NeuralObscura
//
//  Created by Edward Knox on 5/18/17.
//  Copyright © 2017 Paul Bergeron. All rights reserved.
//

import XCTest
import MetalKit
import MetalPerformanceShaders
@testable import NeuralObscura

class NeuralStyleTests: CommandEncoderBaseTest {

    func testImageLoad() {
        let testUrl = Bundle(for: type(of: self))
            .url(forResource: "tubingen", withExtension: "jpg", subdirectory: "testdata")!
        let testImg = UIImage.init(contentsOfFile: testUrl.path)!.toMPSImage(device: device)

        let outImg = UnormToHalfLayer()
            .chain(MPSImageVariable(testImg))
            .forward(commandBuffer: commandBuffer)
        execute()

        let expUrl = Bundle(for: type(of: self))
            .url(forResource: "tubingen", withExtension: "npy", subdirectory: "testdata")!
        let expImg = MPSImage.fromNumpy(expUrl)

        print(outImg)
        print("-----------------------------------------------------------------------")
        print(expImg)
        print("-----------------------------------------------------------------------")
        print(outImg.texture.pixelFormat.description)
        print(outImg.texture.toBase64())
        XCTAssert(testImg.isLossyEqual(image: expImg, precision: 0))
    }

    
    func testStage0() {
        let testUrl = Bundle(for: type(of: self))
            .url(forResource: "tubingen", withExtension: "jpg", subdirectory: "testdata")!
        let testImg = UIImage.init(contentsOfFile: testUrl.path)!.toMPSImage(device: device)
        
        let outImg = UnormToHalfLayer()
            .chain(MPSImageVariable(testImg))
            .forward(commandBuffer: commandBuffer)
        execute()
        
        let expUrl = Bundle(for: type(of: self))
            .url(forResource: "inc_gt_0", withExtension: "npy", subdirectory: "testdata")!
        let expImg = MPSImage.fromNumpy(expUrl)
        
        XCTAssert(outImg.isLossyEqual(image: expImg, precision: 0))
        print(outImg)
        print(expImg)
    }
    
    func testStage1() {
        let testUrl = Bundle(for: type(of: self))
            .url(forResource: "tubingen", withExtension: "jpg", subdirectory: "testdata")!
        let testImg = UIImage.init(contentsOfFile: testUrl.path)!.toMPSImage(device: device)
        
        let c1_w = FileParameterBuffer(modelName: "composition", rawFileName: "c1_W")
        let c1_b = FileParameterBuffer(modelName: "composition", rawFileName: "c1_b")
        
        let preprocess = UnormToHalfLayer()
        let c1 = ConvolutionLayer(
            kernelSize: 9,
            channelsIn: 3,
            channelsOut: 32,
            w: c1_w,
            b: c1_b,
            relu: true,
            padding: 4,
            stride: 1)
        
        let a = preprocess.chain(MPSImageVariable(testImg))
        let b = c1.chain(a)
        let outImg = b.forward(commandBuffer: commandBuffer)
        execute()
        
        let expUrl = Bundle(for: type(of: self))
            .url(forResource: "inc_gt_1", withExtension: "npy", subdirectory: "testdata")!
        let expImg = MPSImage.fromNumpy(expUrl)
        
        XCTAssert(outImg.isLossyEqual(image: expImg, precision: -1))
    }
}
