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
    
    func testGroundTruthConv() throws {
        let testUrl = Bundle(for: type(of: self))
            .url(forResource: "conv_input", withExtension: "npy", subdirectory: "testdata")!
        let testImg = MPSImage.loadFromNumpy(testUrl)
        print("image loaded")
        print(testImg)
        
        print("expected")
        let expUrl = Bundle(for: type(of: self))
            .url(forResource: "conv_expected_output", withExtension: "npy", subdirectory: "testdata")!
        let expImg = MPSImage.loadFromNumpy(expUrl)
        
        let w_pb = FileParameterBuffer(modelName: "composition", rawFileName: "c1_W")
        let b_pb = FileParameterBuffer(modelName: "composition", rawFileName: "c1_b")
        
        let conv = ConvolutionLayer(
            kernelSize: 9,
            channelsIn: 3,
            channelsOut: 32,
            w: w_pb,
            b: b_pb,
            relu: false,
            padding: 4)
        
        let outputImg = conv.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
        execute()
        
        
        /* Verify the result */
        XCTAssert(outputImg.isLossyEqual(image: expImg, precision: -1))
        print(outputImg)
//        print(expImg)
    }
    
    func testGroundTruthConvRelu() {
        let testUrl = Bundle(for: type(of: self))
            .url(forResource: "conv_relu_input", withExtension: "npy", subdirectory: "testdata")!
        let testImg = MPSImage.loadFromNumpy(testUrl)
        
        let w_pb = FileParameterBuffer(modelName: "composition", rawFileName: "c1_W")
        let b_pb = FileParameterBuffer(modelName: "composition", rawFileName: "c1_b")
        
        let conv = ConvolutionLayer(
            kernelSize: 9,
            channelsIn: 3,
            channelsOut: 32,
            w: w_pb,
            b: b_pb,
            relu: true,
            padding: 4)
        
        let outputImg = conv.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
        execute()
        
        let expUrl = Bundle(for: type(of: self))
            .url(forResource: "conv_relu_expected_output", withExtension: "npy", subdirectory: "testdata")!
        let expImg = MPSImage.loadFromNumpy(expUrl)
        
        /* Verify the result */
        XCTAssert(outputImg.isLossyEqual(image: expImg, precision: -1))
    }
    
    func testIdentityNoPadding() {
        let testImg = device.MakeMPSImage(width: 4,
                                          height: 4,
                                          values: [0, 0, 0, 0,
                                                   0, 1, 0, 1,
                                                   0, 0, 1, 0,
                                                   0, 0, 0, 0] as [Float32])
        
        /* Create our CommandEncoder */
        let w_pb = MemoryParameterBuffer([0, 0, 0,
                                          0, 1, 0,
                                          0, 0, 0])
        let b_pb = MemoryParameterBuffer(0)
        let conv = ConvolutionLayer(
            kernelSize: 3,
            channelsIn: 1,
            channelsOut: 1,
            w: w_pb,
            b: b_pb,
            relu: false,
            padding: 0)
        
        let expImg = device.MakeMPSImage(width: 2,
                                         height: 2,
                                         pixelFormat: testTextureFormatR,
                                         values: [1, 0,
                                                  0, 1] as [Float32])
        
        /* Run our test */
        let outputImg = conv.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
        execute()
        
        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }
    
    func testIdentityHalfPadding() {
        let testImg = device.MakeMPSImage(width: 4, height: 4, values: [0, 0, 0, 0,
                                                                        0, 1, 0, 1,
                                                                        0, 0, 1, 0,
                                                                        0, 0, 0, 0] as [Float32])
        
        /* Create our CommandEncoder */
        let w_pb = MemoryParameterBuffer([0, 0, 0,
                                          0, 1, 0,
                                          0, 0, 0])
        let b_pb = MemoryParameterBuffer(0)
        let conv = ConvolutionLayer(
            kernelSize: 3,
            channelsIn: 1,
            channelsOut: 1,
            w: w_pb,
            b: b_pb,
            relu: false,
            padding: 1)
        
        let expImg = device.MakeMPSImage(width: 4,
                                         height: 4,
                                         pixelFormat: testTextureFormatR,
                                         values: [0, 0, 0, 0,
                                                  0, 1, 0, 1,
                                                  0, 0, 1, 0,
                                                  0, 0, 0, 0] as [Float32])
        
        
        /* Run our test */
        let outputImg = conv.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
        execute()
        
        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }

    func testIdentityFullPadding() {
        let testImg = device.MakeMPSImage(width: 4,
                                          height: 4,
                                          values: [[0], [0], [0], [0],
                                                   [0], [1], [0], [1],
                                                   [0], [0], [1], [0],
                                                   [0], [0], [0], [0]] as [[Float32]])
        
        /* Create our CommandEncoder */
        let w_pb = MemoryParameterBuffer([0, 0, 0,
                                          0, 1, 0,
                                          0, 0, 0])
        let b_pb = MemoryParameterBuffer(0)
        let conv = ConvolutionLayer(
            kernelSize: 3,
            channelsIn: 1,
            channelsOut: 1,
            w: w_pb,
            b: b_pb,
            relu: false,
            padding: 2)
        
        let expImg = device.MakeMPSImage(width: 6,
                                         height: 6,
                                         pixelFormat: testTextureFormatR,
                                         values: [[0, 0, 0, 0, 0, 0,
                                                  0, 0, 0, 0, 0, 0,
                                                  0, 0, 1, 0, 1, 0,
                                                  0, 0, 0, 1, 0, 0,
                                                  0, 0, 0, 0, 0, 0,
                                                  0, 0, 0, 0, 0, 0]] as [[Float32]])
        
        
        /* Run our test */
        let outputImg = conv.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
        execute()
        
        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }
    
    func testSumNoPadding() {
        let testImg = device.MakeMPSImage(width: 4, height: 4, values: [[0, 0, 0, 0,
                                                                         0, 1, 0, 1,
                                                                         0, 0, 1, 0,
                                                                         0, 0, 0, 0]] as [[Float32]])
        
        /* Create our CommandEncoder */
        let w_pb = MemoryParameterBuffer([1, 1, 1,
                                          1, 1, 1,
                                          1, 1, 1])
        let b_pb = MemoryParameterBuffer(0)
        let conv = ConvolutionLayer(
            kernelSize: 3,
            channelsIn: 1,
            channelsOut: 1,
            w: w_pb,
            b: b_pb,
            relu: false,
            padding: 0)
        
        let expImg = device.MakeMPSImage(width: 2,
                                         height: 2,
                                         pixelFormat: testTextureFormatR,
                                         values: [2, 3,
                                                  2, 3] as [Float32])
        
        /* Run our test */
        let outputImg = conv.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
        execute()
        
        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }
    
    func testSumHalfPadding() {
        let testImg = device.MakeMPSImage(width: 4,
                                          height: 4,
                                          values: [0, 0, 0, 0,
                                                   0, 1, 0, 1,
                                                   0, 0, 1, 0,
                                                   0, 0, 0, 0] as [Float32])
        /* Create our CommandEncoder */
        let w_pb = MemoryParameterBuffer([1, 1, 1,
                                          1, 1, 1,
                                          1, 1, 1])
        let b_pb = MemoryParameterBuffer(0)
        let conv = ConvolutionLayer(
            kernelSize: 3,
            channelsIn: 1,
            channelsOut: 1,
            w: w_pb,
            b: b_pb,
            relu: false,
            padding: 1)
        
        let expImg = device.MakeMPSImage(width: 4,
                                         height: 4,
                                         pixelFormat: testTextureFormatR,
                                         values: [1, 1, 2, 1,
                                                  1, 2, 3, 2,
                                                  1, 2, 3, 2,
                                                  0, 1, 1, 1] as [Float32])
        
        
        /* Run our test */
        let outputImg = conv.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
        execute()
        
        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }
    
    func testSumFullPadding() {
        let testImg = device.MakeMPSImage(width: 4,
                                          height: 4,
                                          values: [0, 0, 0, 0,
                                                   0, 1, 0, 1,
                                                   0, 0, 1, 0,
                                                   0, 0, 0, 0] as [Float32])
        
        /* Create our CommandEncoder */
        let w_pb = MemoryParameterBuffer([1, 1, 1,
                                          1, 1, 1,
                                          1, 1, 1])
        let b_pb = MemoryParameterBuffer(0)
        let conv = ConvolutionLayer(
            kernelSize: 3,
            channelsIn: 1,
            channelsOut: 1,
            w: w_pb,
            b: b_pb,
            relu: false,
            padding: 2)
        
        let expImg = device.MakeMPSImage(width: 6,
                                         height: 6,
                                         pixelFormat: testTextureFormatR,
                                         values: [0, 0, 0, 0, 0, 0,
                                                  0, 1, 1, 2, 1, 1,
                                                  0, 1, 2, 3, 2, 1,
                                                  0, 1, 2, 3, 2, 1,
                                                  0, 0, 1, 1, 1, 0,
                                                  0, 0, 0, 0, 0, 0] as [Float32])
        
        
        /* Run our test */
        let outputImg = conv.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
        execute()
        
        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }
    
    func testProofOfConcept() {
        /* Create an input test image */
        let testImg = device.MakeMPSImage(width: 4,
                                          height: 4,
                                          values: [0, 0, 0, 0,
                                                   0, 3, 3, 0,
                                                   0, 6, 1, 0,
                                                   0, 0, 0, 0] as [Float32])
        /* Create our CommandEncoder */
        let w: [Float] = [1, 1,
                          1, 1]
        let b: [Float] = [0]
        let convDesc = MPSCNNConvolutionDescriptor(
            kernelWidth: 2,
            kernelHeight: 2,
            inputFeatureChannels: 1,
            outputFeatureChannels: 1,
            neuronFilter: nil)
        let conv = MPSCNNConvolution(
            device: device,
            convolutionDescriptor: convDesc,
            kernelWeights: w,
            biasTerms: b,
            flags: MPSCNNConvolutionFlags.none)
        conv.edgeMode = .zero
        conv.offset = MPSOffset(x: 0, y: 0, z: 0)
        conv.clipRect.size = MTLSizeMake(testImg.width + 1, testImg.height + 1, 1)
        
        /* Create an expected output image */
        let expImg = device.MakeMPSImage(width: 5,
                                         height: 5,
                                         pixelFormat: testTextureFormatR,
                                         values: [0, 0, 0, 0, 0,
                                                  0, 3, 6, 3, 0,
                                                  0, 9, 13, 4, 0,
                                                  0, 6,  7, 1, 0,
                                                  0, 0, 0, 0, 0] as [Float32])

        /*  Create an output image */
        let outputImg = MPSImage(
            device: device,
            imageDescriptor: MPSImageDescriptor(
                channelFormat: textureFormat,
                width: 5,
                height: 5,
                featureChannels: 1))
        
        /* Run our test */
        conv.encode(commandBuffer: commandBuffer, sourceImage: testImg, destinationImage: outputImg)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }
    
}
