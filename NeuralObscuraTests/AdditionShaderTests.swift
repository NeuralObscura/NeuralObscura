//
//  ResidualBlockSummationShaderTests.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 11/5/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import XCTest
import MetalKit
import MetalPerformanceShaders
@testable import NeuralObscura

class AdditionShaderTests: CommandEncoderBaseTest {
    func testResidualBlockSummationShader() {
        let testImg1 = device.MakeTestMPSImage(width: 2,
                                               height: 2,
                                               featureChannels: 4,
                                               pixelFormat: MTLPixelFormat.rgba16Float,
                                               values: [[1,2,3,4], [4,3,2,1],
                                                        [3,4,2,1], [2,1,3,4]])

        let testImg2 = device.MakeTestMPSImage(width: 2,
                                               height: 2,
                                               featureChannels: 4,
                                               pixelFormat: MTLPixelFormat.rgba16Float,
                                               values: [[1,2,3,4], [4,3,2,1],
                                                        [3,4,2,1], [2,1,3,4]])

        let outputImg = device.MakeTestMPSImage(width: 2,
                                                height: 2,
                                                featureChannels: 4,
                                                pixelFormat: MTLPixelFormat.rgba16Float,
                                                values: [[0,0,0,0], [0,0,0,0],
                                                         [0,0,0,0], [0,0,0,0]])

        /* Create our CommandEncoder*/
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.setComputePipelineState(ShaderRegistry.getOrLoad(name: "add"))
        encoder.setTexture(testImg1.texture, at: 0)
        encoder.setTexture(testImg2.texture, at: 1)
        encoder.setTexture(outputImg.texture, at: 2)
        let threadsPerGroup = MTLSizeMake(1, 1, 1)
        let threadGroups = MTLSizeMake(outputImg.texture.width,
                                       outputImg.texture.height, 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        /* Run our test */
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let expImg = device.MakeTestMPSImage(width: 2,
                                             height: 2,
                                             featureChannels: 4,
                                             pixelFormat: MTLPixelFormat.rgba16Float,
                                             values: [[2,4,6,8], [8,6,4,2],
                                                      [6,8,4,2], [4,2,6,8]])

        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }
}
