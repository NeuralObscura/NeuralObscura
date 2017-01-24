//
//  MPSImageExtensionsTests.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 1/18/17.
//  Copyright © 2017 Paul Bergeron. All rights reserved.
//

import XCTest
import MetalKit
import MetalPerformanceShaders
@testable import NeuralObscura

class MPSImageExtensionsTests: CommandEncoderBaseTest {

    func testLoadFromNumpy() {
        let url = Bundle(for: type(of: self))
            .url(forResource: "test_loadFromNumpy_data", withExtension: "npy", subdirectory: "testdata")!
        let outputImg = MPSImage.loadFromNumpy(url, destinationPixelFormat: .r32Float)

        let expImg = [1.1, 2.2, 3.3, 4.4] as [Float32]

        XCTAssert(outputImg.isLossyEqual(expImg, precision: 2))
    }
}
