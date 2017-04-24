//
//  FileParameterBufferTests.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 11/9/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//


import XCTest
import MetalKit
import MetalPerformanceShaders
@testable import NeuralObscura

class FileParameterBufferTests: CommandEncoderBaseTest {
    let precision: Float = 10000

    func testFileParameterBufferLoadsBias() {
        let b_pb = FileParameterBuffer(modelName: "composition", rawFileName: "c1_b")

        let expectedBias = [ 3.77813292,  1.9199549 ,  3.62352777,  2.59458852,  5.0819459 ,
                             2.89543724, -0.30800632,  3.14301109,  5.22599697,  3.09896779,
                             4.54812527,  1.1446383 ,  0.11570477,  0.90781438,  0.75834346,
                             2.28524208, -0.60550272,  0.60350686,  1.86674798,  1.0393852 ,
                             3.28579903,  2.44887495,  2.57894325, -0.11285205,  4.4115901 ,
                             3.08820701,  2.97502899,  0.37210375,  2.28866887,  0.50753558,
                             1.10829115, -0.06531702] as [Float32]

        XCTAssertEqual(b_pb.length / MemoryLayout<Float32>.size, expectedBias.count)

        let buffer = UnsafeBufferPointer<Float32>(start: b_pb.pointer, count: b_pb.length / MemoryLayout<Float32>.size)

        buffer.enumerated().forEach { [unowned self] (idx, e) in
            let actual   = Int(e * self.precision)
            let expected = Int(expectedBias[idx] * self.precision)
            XCTAssertEqual(actual, expected)
        }
    }
}
