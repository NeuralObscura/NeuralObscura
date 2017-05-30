//
//  TanhAdjustmentLayerTests.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 11/6/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import XCTest
import MetalKit
import MetalPerformanceShaders
@testable import NeuralObscura

class TanhAdjustmentLayerTests: CommandEncoderBaseTest {
    func testTanhAdjustmentLayer() {
        let testImg = device.makeMPSImage(width: 2,
                                          height: 2,
                                          values: [[-25, -25,
                                                    -25, -25],
                                                   [-25, -25,
                                                    -25, -25],
                                                   [-25, -25,
                                                    -25, -25],
                                                   [-25, -25,
                                                    -25, -25]])
        let tanhAdj = TanhAdjustmentLayer()
        
        let outputImg = tanhAdj.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
        execute()

        let expImg = device.makeMPSImage(width: 2,
                                         height: 2,
                                         values: [[0,0,
                                                   0,0],
                                                  [0,0,
                                                   0,0],
                                                  [0,0,
                                                   0,0],
                                                  [255,255,
                                                   255,255]])
        XCTAssertEqual(outputImg, expImg)
    }
}
