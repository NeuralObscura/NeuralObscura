//
//  DeconvPart2Tests.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 2/15/17.
//  Copyright © 2017 Paul Bergeron. All rights reserved.
//

import XCTest
import MetalKit
import MetalPerformanceShaders
@testable import NeuralObscura

class DeconvPart2Tests: CommandEncoderBaseTest {
    func testDeconvPart2Shader() {
        let bytes : [Float32] =
            [ -1,   0,   1,   2,  -2,   0,   2,   4,  -3,   0,   3,   6,  -4,   0,   4,
              8,  -5,   0,   5,  10,  -6,   0,   6,  12,  -7,   0,   7,  14,  -8,   0,
              8,  16,  -9,   0,   9,  18, -10,   0,  10,  20, -11,   0,  11,  22, -12,
              0,  12,  24, -13,   0,  13,  26, -14,   0,  14,  28, -15,   0,  15,  30,
              -16,   0,  16,  32]

        let inputBuffer = device.makeBuffer(bytes: bytes, length: bytes.count, options: MTLResourceOptions.storageModeShared)

        let outputImg = device.MakeMPSImage(width: 5,
                                              height: 5,
                                              featureChannels: 1,
                                              pixelFormat: .r32Float,
                                              textureType: .type2D,
                                              values: [0,0,0,0,0,
                                                       0,0,0,0,0,
                                                       0,0,0,0,0,
                                                       0,0,0,0,0,
                                                       0,0,0,0,0])

        let encoder = commandBuffer.makeComputeCommandEncoder()
        let state = ShaderRegistry.getOrLoad(name: "col2im")
        encoder.setComputePipelineState(state)
        encoder.setBuffer(inputBuffer, offset: 0, at: 0)
        encoder.setTexture(outputImg.texture, at: 1)

        let params = [
            UInt32(1),  // nc_out
            UInt32(2),  // nh
            UInt32(2),  // nw
            UInt32(64), //inputSize
            UInt32(5),  // nkh
            UInt32(5),  // nkw
            UInt32(5),  // output row width
            UInt32(1),  // stride
            UInt32(0),  //padding
            ] as [UInt32]

        let paramsBuffer = device.makeBuffer(bytes: params,
                                             length: params.count * MemoryLayout<UInt32>.size,
                                             options: .cpuCacheModeWriteCombined)

        encoder.setBuffer(paramsBuffer, offset:0, at: 2)

        let threadsPerGroup = MTLSizeMake(1, 1, 1)
        let threadGroups = MTLSizeMake(bytes.count, 1, 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        /* Run our test */
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        print(outputImg)
        print("==========================================================")
    }
}
