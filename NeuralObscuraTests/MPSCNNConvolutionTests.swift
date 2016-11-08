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

class MPSCNNConvolutionTests: CommandEncoderBaseTest {
    
    func testProofOfConcept() {
        /* Create an input test image */
        let testImg = device.MakeTestMPSImage(width: 4, height: 4, values: [0, 0, 0, 0,
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
        let expImg = device.MakeTestMPSImage(width: 5, height: 5, values: [0, 0, 0, 0, 0,
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
