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
    
//    func testAll() {
//        let debug = false
//        let modelName = "composition"
//        var modelParams = [String: ParameterBuffer]()
//        
//        let src: MPSImageVariable
//        let c1, c2, c3: ConvolutionLayer
//        let b1, b2, b3, b4, b5: BatchNormalizationNonTestLayer
//        let r1, r2, r3, r4, r5: ResidualBlock
//        let d1, d2, d3: DeconvolutionLayerV2
//        let tanhAdj: TanhAdjustmentLayer
//        let bgra_to_brga: BGRAToBRGALayer
//        let model: AnyCommandEncoder<MPSImage>
//        
//        /* Load model parameters */
//        modelParams["r4_b1_mean"] = FileParameterBuffer(modelName: modelName, rawFileName: "r4_b1_mean")
//        //r4_b1_mean shape = (128,)
//        modelParams["r4_b1_stddev"] = FileParameterBuffer(modelName: modelName, rawFileName: "r4_b1_stddev")
//        //r4_b1_stddev shape = (128,)
//        modelParams["r4_b2_mean"] = FileParameterBuffer(modelName: modelName, rawFileName: "r4_b2_mean")
//        //r4_b2_mean shape = (128,)
//        modelParams["r4_b2_stddev"] = FileParameterBuffer(modelName: modelName, rawFileName: "r4_b2_stddev")
//        //r4_b2_stddev shape = (128,)
//        modelParams["r4_c2_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "r4_c2_W")
//        //r4_c2_W shape = (128, 3, 3, 128)
//        modelParams["r4_c2_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "r4_c2_b")
//        //r4_c2_b shape = (128,)
//        modelParams["r4_c1_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "r4_c1_W")
//        //r4_c1_W shape = (128, 3, 3, 128)
//        modelParams["r4_c1_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "r4_c1_b")
//        //r4_c1_b shape = (128,)
//        modelParams["r4_b1_gamma"] = FileParameterBuffer(modelName: modelName, rawFileName: "r4_b1_gamma")
//        //r4_b1_gamma shape = (128,)
//        modelParams["r4_b1_beta"] = FileParameterBuffer(modelName: modelName, rawFileName: "r4_b1_beta")
//        //r4_b1_beta shape = (128,)
//        modelParams["r4_b2_gamma"] = FileParameterBuffer(modelName: modelName, rawFileName: "r4_b2_gamma")
//        //r4_b2_gamma shape = (128,)
//        modelParams["r4_b2_beta"] = FileParameterBuffer(modelName: modelName, rawFileName: "r4_b2_beta")
//        //r4_b2_beta shape = (128,)
//        modelParams["r5_b1_mean"] = FileParameterBuffer(modelName: modelName, rawFileName: "r5_b1_mean")
//        //r5_b1_mean shape = (128,)
//        modelParams["r5_b1_stddev"] = FileParameterBuffer(modelName: modelName, rawFileName: "r5_b1_stddev")
//        //r5_b1_stddev shape = (128,)
//        modelParams["r5_b2_mean"] = FileParameterBuffer(modelName: modelName, rawFileName: "r5_b2_mean")
//        //r5_b2_mean shape = (128,)
//        modelParams["r5_b2_stddev"] = FileParameterBuffer(modelName: modelName, rawFileName: "r5_b2_stddev")
//        //r5_b2_stddev shape = (128,)
//        modelParams["r5_c2_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "r5_c2_W")
//        //r5_c2_W shape = (128, 3, 3, 128)
//        modelParams["r5_c2_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "r5_c2_b")
//        //r5_c2_b shape = (128,)
//        modelParams["r5_c1_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "r5_c1_W")
//        //r5_c1_W shape = (128, 3, 3, 128)
//        modelParams["r5_c1_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "r5_c1_b")
//        //r5_c1_b shape = (128,)
//        modelParams["r5_b1_gamma"] = FileParameterBuffer(modelName: modelName, rawFileName: "r5_b1_gamma")
//        //r5_b1_gamma shape = (128,)
//        modelParams["r5_b1_beta"] = FileParameterBuffer(modelName: modelName, rawFileName: "r5_b1_beta")
//        //r5_b1_beta shape = (128,)
//        modelParams["r5_b2_gamma"] = FileParameterBuffer(modelName: modelName, rawFileName: "r5_b2_gamma")
//        //r5_b2_gamma shape = (128,)
//        modelParams["r5_b2_beta"] = FileParameterBuffer(modelName: modelName, rawFileName: "r5_b2_beta")
//        //r5_b2_beta shape = (128,)
//        modelParams["r1_b1_mean"] = FileParameterBuffer(modelName: modelName, rawFileName: "r1_b1_mean")
//        //r1_b1_mean shape = (128,)
//        modelParams["r1_b1_stddev"] = FileParameterBuffer(modelName: modelName, rawFileName: "r1_b1_stddev")
//        //r1_b1_stddev shape = (128,)
//        modelParams["r1_b2_mean"] = FileParameterBuffer(modelName: modelName, rawFileName: "r1_b2_mean")
//        //r1_b2_mean shape = (128,)
//        modelParams["r1_b2_stddev"] = FileParameterBuffer(modelName: modelName, rawFileName: "r1_b2_stddev")
//        //r1_b2_stddev shape = (128,)
//        modelParams["r1_c2_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "r1_c2_W")
//        //r1_c2_W shape = (128, 3, 3, 128)
//        modelParams["r1_c2_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "r1_c2_b")
//        //r1_c2_b shape = (128,)
//        modelParams["r1_c1_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "r1_c1_W")
//        //r1_c1_W shape = (128, 3, 3, 128)
//        modelParams["r1_c1_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "r1_c1_b")
//        //r1_c1_b shape = (128,)
//        modelParams["r1_b1_gamma"] = FileParameterBuffer(modelName: modelName, rawFileName: "r1_b1_gamma")
//        //r1_b1_gamma shape = (128,)
//        modelParams["r1_b1_beta"] = FileParameterBuffer(modelName: modelName, rawFileName: "r1_b1_beta")
//        //r1_b1_beta shape = (128,)
//        modelParams["r1_b2_gamma"] = FileParameterBuffer(modelName: modelName, rawFileName: "r1_b2_gamma")
//        //r1_b2_gamma shape = (128,)
//        modelParams["r1_b2_beta"] = FileParameterBuffer(modelName: modelName, rawFileName: "r1_b2_beta")
//        //r1_b2_beta shape = (128,)
//        modelParams["r2_b1_mean"] = FileParameterBuffer(modelName: modelName, rawFileName: "r2_b1_mean")
//        //r2_b1_mean shape = (128,)
//        modelParams["r2_b1_stddev"] = FileParameterBuffer(modelName: modelName, rawFileName: "r2_b1_stddev")
//        //r2_b1_stddev shape = (128,)
//        modelParams["r2_b2_mean"] = FileParameterBuffer(modelName: modelName, rawFileName: "r2_b2_mean")
//        //r2_b2_mean shape = (128,)
//        modelParams["r2_b2_stddev"] = FileParameterBuffer(modelName: modelName, rawFileName: "r2_b2_stddev")
//        //r2_b2_stddev shape = (128,)
//        modelParams["r2_c2_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "r2_c2_W")
//        //r2_c2_W shape = (128, 3, 3, 128)
//        modelParams["r2_c2_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "r2_c2_b")
//        //r2_c2_b shape = (128,)
//        modelParams["r2_c1_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "r2_c1_W")
//        //r2_c1_W shape = (128, 3, 3, 128)
//        modelParams["r2_c1_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "r2_c1_b")
//        //r2_c1_b shape = (128,)
//        modelParams["r2_b1_gamma"] = FileParameterBuffer(modelName: modelName, rawFileName: "r2_b1_gamma")
//        //r2_b1_gamma shape = (128,)
//        modelParams["r2_b1_beta"] = FileParameterBuffer(modelName: modelName, rawFileName: "r2_b1_beta")
//        //r2_b1_beta shape = (128,)
//        modelParams["r2_b2_gamma"] = FileParameterBuffer(modelName: modelName, rawFileName: "r2_b2_gamma")
//        //r2_b2_gamma shape = (128,)
//        modelParams["r2_b2_beta"] = FileParameterBuffer(modelName: modelName, rawFileName: "r2_b2_beta")
//        //r2_b2_beta shape = (128,)
//        modelParams["r3_b1_mean"] = FileParameterBuffer(modelName: modelName, rawFileName: "r3_b1_mean")
//        //r3_b1_mean shape = (128,)
//        modelParams["r3_b1_stddev"] = FileParameterBuffer(modelName: modelName, rawFileName: "r3_b1_stddev")
//        //r3_b1_stddev shape = (128,)
//        modelParams["r3_b2_mean"] = FileParameterBuffer(modelName: modelName, rawFileName: "r3_b2_mean")
//        //r3_b2_mean shape = (128,)
//        modelParams["r3_b2_stddev"] = FileParameterBuffer(modelName: modelName, rawFileName: "r3_b2_stddev")
//        //r3_b2_stddev shape = (128,)
//        modelParams["r3_c2_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "r3_c2_W")
//        //r3_c2_W shape = (128, 3, 3, 128)
//        modelParams["r3_c2_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "r3_c2_b")
//        //r3_c2_b shape = (128,)
//        modelParams["r3_c1_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "r3_c1_W")
//        //r3_c1_W shape = (128, 3, 3, 128)
//        modelParams["r3_c1_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "r3_c1_b")
//        //r3_c1_b shape = (128,)
//        modelParams["r3_b1_gamma"] = FileParameterBuffer(modelName: modelName, rawFileName: "r3_b1_gamma")
//        //r3_b1_gamma shape = (128,)
//        modelParams["r3_b1_beta"] = FileParameterBuffer(modelName: modelName, rawFileName: "r3_b1_beta")
//        //r3_b1_beta shape = (128,)
//        modelParams["r3_b2_gamma"] = FileParameterBuffer(modelName: modelName, rawFileName: "r3_b2_gamma")
//        //r3_b2_gamma shape = (128,)
//        modelParams["r3_b2_beta"] = FileParameterBuffer(modelName: modelName, rawFileName: "r3_b2_beta")
//        //r3_b2_beta shape = (128,)
//        modelParams["b4_mean"] = FileParameterBuffer(modelName: modelName, rawFileName: "b4_mean")
//        //b4_mean shape = (64,)
//        modelParams["b4_stddev"] = FileParameterBuffer(modelName: modelName, rawFileName: "b4_stddev")
//        //b4_stddev shape = (64,)
//        modelParams["b4_gamma"] = FileParameterBuffer(modelName: modelName, rawFileName: "b4_gamma")
//        //b4_gamma shape = (64,)
//        modelParams["b4_beta"] = FileParameterBuffer(modelName: modelName, rawFileName: "b4_beta")
//        //b4_beta shape = (64,)
//        modelParams["b5_mean"] = FileParameterBuffer(modelName: modelName, rawFileName: "b5_mean")
//        //b5_mean shape = (32,)
//        modelParams["b5_stddev"] = FileParameterBuffer(modelName: modelName, rawFileName: "b5_stddev")
//        //b5_stddev shape = (32,)
//        modelParams["b5_gamma"] = FileParameterBuffer(modelName: modelName, rawFileName: "b5_gamma")
//        //b5_gamma shape = (32,)
//        modelParams["b5_beta"] = FileParameterBuffer(modelName: modelName, rawFileName: "b5_beta")
//        //b5_beta shape = (32,)
//        modelParams["b1_mean"] = FileParameterBuffer(modelName: modelName, rawFileName: "b1_mean")
//        //b1_mean shape = (32,)
//        modelParams["b1_stddev"] = FileParameterBuffer(modelName: modelName, rawFileName: "b1_stddev")
//        //b1_stddev shape = (32,)
//        modelParams["b1_gamma"] = FileParameterBuffer(modelName: modelName, rawFileName: "b1_gamma")
//        //b1_gamma shape = (32,)
//        modelParams["b1_beta"] = FileParameterBuffer(modelName: modelName, rawFileName: "b1_beta")
//        //b1_beta shape = (32,)
//        modelParams["b2_mean"] = FileParameterBuffer(modelName: modelName, rawFileName: "b2_mean")
//        //b2_mean shape = (64,)
//        modelParams["b2_stddev"] = FileParameterBuffer(modelName: modelName, rawFileName: "b2_stddev")
//        //b2_stddev shape = (64,)
//        modelParams["b2_gamma"] = FileParameterBuffer(modelName: modelName, rawFileName: "b2_gamma")
//        //b2_gamma shape = (64,)
//        modelParams["b2_beta"] = FileParameterBuffer(modelName: modelName, rawFileName: "b2_beta")
//        //b2_beta shape = (64,)
//        modelParams["b3_mean"] = FileParameterBuffer(modelName: modelName, rawFileName: "b3_mean")
//        //b3_mean shape = (128,)
//        modelParams["b3_stddev"] = FileParameterBuffer(modelName: modelName, rawFileName: "b3_stddev")
//        //b3_stddev shape = (128,)
//        modelParams["b3_gamma"] = FileParameterBuffer(modelName: modelName, rawFileName: "b3_gamma")
//        //b3_gamma shape = (128,)
//        modelParams["b3_beta"] = FileParameterBuffer(modelName: modelName, rawFileName: "b3_beta")
//        //b3_beta shape = (128,)
//        modelParams["c3_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "c3_W")
//        //c3_W shape = (128, 4, 4, 64)
//        modelParams["c3_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "c3_b")
//        //c3_b shape = (128,)
//        modelParams["c2_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "c2_W")
//        //c2_W shape = (64, 4, 4, 32)
//        modelParams["c2_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "c2_b")
//        //c2_b shape = (64,)
//        modelParams["c1_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "c1_W")
//        //c1_W shape = (32, 9, 9, 3)
//        modelParams["c1_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "c1_b")
//        //c1_b shape = (32,)
//        modelParams["d2_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "d2_W")
//        //d2_W shape = (64, 4, 4, 32)
//        modelParams["d2_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "d2_b")
//        //d2_b shape = (32,)
//        modelParams["d3_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "d3_W")
//        //d3_W shape = (32, 9, 9, 3)
//        modelParams["d3_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "d3_b")
//        //d3_b shape = (3,)
//        modelParams["d1_W"] = FileParameterBuffer(modelName: modelName, rawFileName: "d1_W")
//        //d1_W shape = (128, 4, 4, 64)
//        modelParams["d1_b"] = FileParameterBuffer(modelName: modelName, rawFileName: "d1_b")
//        //d1_b shape = (64,)
//        
//        /* Init model encoders */
//        src = MPSImageVariable()
//        
//        bgra_to_brga = BGRAToBRGALayer()
//        
//        // c1=L.Convolution2D(3, 32, 9, stride=1, pad=4),
//        c1 = ConvolutionLayer(
//            kernelSize: 9,
//            channelsIn: 3,
//            channelsOut: 32,
//            w: modelParams["c1_W"]!,
//            b: modelParams["c1_b"]!,
//            relu: true,
//            padding: 4,
//            stride: 1)
//        
//        // b1=L.BatchNormalization(32),
//        b1 = BatchNormalizationNonTestLayer(
//            beta: modelParams["b1_beta"]!,
//            gamma: modelParams["b1_gamma"]!)
//        
//        // c2=L.Convolution2D(32, 64, 4, stride=2, pad=1),
//        c2 = ConvolutionLayer(
//            kernelSize: 4,
//            channelsIn: 32,
//            channelsOut: 64,
//            w: modelParams["c2_W"]!,
//            b: modelParams["c2_b"]!,
//            relu: true,
//            padding: 1,
//            stride: 2)
//        
//        // b2=L.BatchNormalization(64),
//        b2 = BatchNormalizationNonTestLayer(
//            beta: modelParams["b2_beta"]!,
//            gamma: modelParams["b2_gamma"]!)
//        
//        // c3=L.Convolution2D(64, 128, 4,stride=2, pad=1),
//        c3 = ConvolutionLayer(
//            kernelSize: 4,
//            channelsIn: 64,
//            channelsOut: 128,
//            w: modelParams["c3_W"]!,
//            b: modelParams["c3_b"]!,
//            relu: true,
//            padding: 1,
//            stride: 2)
//        
//        // b3=L.BatchNormalization(128),
//        b3 = BatchNormalizationNonTestLayer(
//            beta: modelParams["b3_beta"]!,
//            gamma: modelParams["b3_gamma"]!)
//        
//        // r1=ResidualBlock(128, 128),
//        r1 = ResidualBlock(
//            modelParams: modelParams,
//            blockName: "r1",
//            channelsIn: 128,
//            channelsOut: 128)
//        
//        // r2=ResidualBlock(128, 128),
//        r2 = ResidualBlock(
//            modelParams: modelParams,
//            blockName: "r2",
//            channelsIn: 128,
//            channelsOut: 128)
//        
//        // r3=ResidualBlock(128, 128),
//        r3 = ResidualBlock(
//            modelParams: modelParams,
//            blockName: "r3",
//            channelsIn: 128,
//            channelsOut: 128)
//        
//        // r4=ResidualBlock(128, 128),
//        r4 = ResidualBlock(
//            modelParams: modelParams,
//            blockName: "r4",
//            channelsIn: 128,
//            channelsOut: 128)
//        
//        // r5=ResidualBlock(128, 128),
//        r5 = ResidualBlock(
//            modelParams: modelParams,
//            blockName: "r5",
//            channelsIn: 128,
//            channelsOut: 128)
//        
//        // d1=L.Deconvolution2D(128, 64, 4, stride=2, pad=1),
//        d1 = DeconvolutionLayerV2(
//            kernelSize: 4,
//            channelsIn: 128,
//            channelsOut: 64,
//            w: modelParams["d1_W"]!,
//            b: modelParams["d1_b"]!,
//            relu: true,
//            padding: 1,
//            stride: 2)
//        
//        // b4=L.BatchNormalization(64),
//        b4 = BatchNormalizationNonTestLayer(
//            beta: modelParams["b4_beta"]!,
//            gamma: modelParams["b4_gamma"]!)
//        
//        // d2=L.Deconvolution2D(64, 32, 4, stride=2, pad=1),
//        d2 = DeconvolutionLayerV2(
//            kernelSize: 4,
//            channelsIn: 64,
//            channelsOut: 32,
//            w: modelParams["d2_W"]!,
//            b: modelParams["d2_b"]!,
//            relu: true,
//            padding: 1,
//            stride: 2)
//        
//        // b5=L.BatchNormalization(32),
//        b5 = BatchNormalizationNonTestLayer(
//            beta: modelParams["b5_beta"]!,
//            gamma: modelParams["b5_gamma"]!)
//        
//        // d3=L.Deconvolution2D(32, 3, 9, stride=1, pad=4),
//        d3 = DeconvolutionLayerV2(
//            kernelSize: 9,
//            channelsIn: 32,
//            channelsOut: 3,
//            w: modelParams["d3_W"]!,
//            b: modelParams["d3_b"]!,
//            relu: false,
//            padding: 4,
//            stride: 1)
//        
//        tanhAdj = TanhAdjustmentLayer()
//        
//        /* Chain model encoders together */
//        var h: AnyCommandEncoder<MPSImage>
//        let l0 = bgra_to_brga.chain(src)
//        
//        // h = self.b1(F.elu(self.c1(top)), test=test)
//        let l1 = c1.chain(l0)
//        let l2 = b1.chain(l1)
//        
//        // h = self.b2(F.elu(self.c2(h)), test=test)
//        let l3 = c2.chain(l2)
//        let l4 = b2.chain(l3)
//        
//        // h = self.b3(F.elu(self.c3(h)), test=test)
//        let l5 = c3.chain(l4)
//        let l6 = b3.chain(l5)
//        
//        // h = self.r1(h, test=test)
//        let l7 = r1.chain(l6)
//        
//        // h = self.r2(h, test=test)
//        let l8 = r2.chain(l7)
//        
//        // h = self.r3(h, test=test)
//        let l9 = r3.chain(l8)
//        
//        // h = self.r4(h, test=test)
//        let l10 = r4.chain(l9)
//        
//        // h = self.r5(h, test=test)
//        let l11 = r5.chain(l10)
//        
//        // h = self.b4(F.elu(self.d1(h)), test=test)
//        let l12 = d1.chain(l11)
//        let l13 = b4.chain(l12)
//        
//        // h = self.b5(F.elu(self.d2(h)), test=test)
//        let l14 = d2.chain(l13)
//        let l15 = b5.chain(l14)
//        
//        // y = self.d3(h)
//        let l16 = d3.chain(l15)
//        
//        // return (F.tanh(y)+1)*127.5
//        let l17 = tanhAdj.chain(l16)
//        
//        let bundle = Bundle(for: type(of: self))
//        let testUrl = bundle.url(forResource: "tubingen", withExtension: "jpg", subdirectory: "testdata")!
//        let testImg = UIImage.init(contentsOfFile: testUrl.path)!.toMPSImage(device: device)
//        
//        let gt0 = MPSImage.fromNumpy(bundle.url(forResource: "inc_gt_0", withExtension: "npy", subdirectory: "testdata")!)
//        let gt1 = MPSImage.fromNumpy(bundle.url(forResource: "inc_gt_1", withExtension: "npy", subdirectory: "testdata")!)
//        let gt2 = MPSImage.fromNumpy(bundle.url(forResource: "inc_gt_2", withExtension: "npy", subdirectory: "testdata")!)
//        let gt3 = MPSImage.fromNumpy(bundle.url(forResource: "inc_gt_3", withExtension: "npy", subdirectory: "testdata")!)
//        let gt4 = MPSImage.fromNumpy(bundle.url(forResource: "inc_gt_4", withExtension: "npy", subdirectory: "testdata")!)
//        let gt5 = MPSImage.fromNumpy(bundle.url(forResource: "inc_gt_5", withExtension: "npy", subdirectory: "testdata")!)
//        let gt6 = MPSImage.fromNumpy(bundle.url(forResource: "inc_gt_6", withExtension: "npy", subdirectory: "testdata")!)
//        let gt7 = MPSImage.fromNumpy(bundle.url(forResource: "inc_gt_7", withExtension: "npy", subdirectory: "testdata")!)
//        let gt8 = MPSImage.fromNumpy(bundle.url(forResource: "inc_gt_8", withExtension: "npy", subdirectory: "testdata")!)
//        let gt9 = MPSImage.fromNumpy(bundle.url(forResource: "inc_gt_9", withExtension: "npy", subdirectory: "testdata")!)
//        let gt10 = MPSImage.fromNumpy(bundle.url(forResource: "inc_gt_10", withExtension: "npy", subdirectory: "testdata")!)
//        let gt11 = MPSImage.fromNumpy(bundle.url(forResource: "inc_gt_11", withExtension: "npy", subdirectory: "testdata")!)
//        let gt12 = MPSImage.fromNumpy(bundle.url(forResource: "inc_gt_12", withExtension: "npy", subdirectory: "testdata")!)
//        let gt13 = MPSImage.fromNumpy(bundle.url(forResource: "inc_gt_13", withExtension: "npy", subdirectory: "testdata")!)
//        let gt14 = MPSImage.fromNumpy(bundle.url(forResource: "inc_gt_14", withExtension: "npy", subdirectory: "testdata")!)
//        let gt15 = MPSImage.fromNumpy(bundle.url(forResource: "inc_gt_15", withExtension: "npy", subdirectory: "testdata")!)
//        let gt16 = MPSImage.fromNumpy(bundle.url(forResource: "inc_gt_16", withExtension: "npy", subdirectory: "testdata")!)
//        let gt17 = MPSImage.fromNumpy(bundle.url(forResource: "inc_gt_17", withExtension: "npy", subdirectory: "testdata")!)
//        
//        src.setValue(testImg)
//        
//        let outputImage = l17.forward(commandBuffer: commandBuffer)
//        execute()
//        
//    }
}
