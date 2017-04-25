//
//  RGBAToBRGALayerTests.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 11/26/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import XCTest
import MetalKit
import MetalPerformanceShaders
@testable import NeuralObscura

class RGBAToBRGALayerTests: CommandEncoderBaseTest {
    func testRGBAToBRGALayer() {
        let testImg = device.makeMPSImage(width: 2,
                                          height: 2,
                                          values: [[1,4,3,2],
                                                   [2,3,4,1],
                                                   [3,2,2,3],
                                                   [4,1,1,4]])

        let tanhAdj = RGBAToBRGALayer()

        let outputImg = tanhAdj.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
        execute()

        let expImg = device.makeMPSImage(width: 2,
                                         height: 2,
                                         values: [[3,2,2,3],
                                                  [1,4,4,1],
                                                  [2,3,3,2],
                                                  [4,1,1,4]])

        XCTAssertEqual(outputImg, expImg)
    }
}
