//
//  DeconvolutionLayerTests.swift
//  NeuralObscura
//
//  Created by Edward Knox on 11/4/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import XCTest
import MetalKit
import MetalPerformanceShaders
@testable import NeuralObscura

class DeconvolutionBlockTests: CommandEncoderBaseTest {
    
    func testGroundTruthTensorDot() {
        let testUrl = Bundle(for: type(of: self))
            .url(forResource: "deconv_input", withExtension: "npy", subdirectory: "testdata")!
        let testImg = MPSImage.loadFromNumpy(testUrl)
        
        let w_pb = FileParameterBuffer(modelName: "composition", rawFileName: "d1_W")
        
        let tensorDot = TensorDotLayer(
            kernelSize: 4,
            channelsIn: 128,
            channelsOut: 64,
            w: w_pb)
        
        let outputBuf = tensorDot.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
        execute()
        let outputBufCount = outputBuf.length / MemoryLayout<Float32>.size
        let outputBufTypedPtr = outputBuf.contents().bindMemory(to: Float32.self, capacity: outputBufCount)
        let outputBuffer = UnsafeBufferPointer.init(start: outputBufTypedPtr, count: outputBufCount)
        
        let expUrl = Bundle(for: type(of: self))
            .url(forResource: "tensordot_expected_output", withExtension: "dat", subdirectory: "testdata")!
        let expData = try! Data.init(contentsOf: expUrl)
        let expBufCount = expData.count / MemoryLayout<Float32>.size
        let expBufPtr = UnsafeMutableRawPointer.allocate(bytes: expData.count, alignedTo: MemoryLayout<Float32>.alignment)
        let expBufTypedPtr = expBufPtr.bindMemory(to: Float32.self, capacity: expBufCount)
        let expBuffer = UnsafeMutableBufferPointer.init(start: expBufTypedPtr, count: expBufCount)
        expData.copyBytes(to: expBuffer)
        for (i, (a,   b)) in zip(0..<outputBuffer.count, zip(outputBuffer, expBuffer)) {
            let diff = abs(a - b)
            XCTAssertLessThan(diff, 0.01, "Index \(i) expected \(b) received \(a)")
        }
    }
    
//    func testGroundTruthDeconv() {
//        let testUrl = Bundle(for: type(of: self))
//            .url(forResource: "deconv_input", withExtension: "npy", subdirectory: "testdata")!
//        let testImg = MPSImage.loadFromNumpy(testUrl, destinationPixelFormat: testTextureFormatRGBA)
//        
//        let w_pb = FileParameterBuffer(modelName: "composition", rawFileName: "d1_W")
//        let b_pb = FileParameterBuffer(modelName: "composition", rawFileName: "d1_b")
//        
//        let deconv = DeconvolutionBlock(
//            kernelSize: 4,
//            channelsIn: 128,
//            channelsOut: 64,
//            w: w_pb,
//            b: b_pb,
//            relu: false,
//            padding: 1,
//            stride: 2)
//        
//        let outputImg = deconv.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
//        execute()
//        
//        let expUrl = Bundle(for: type(of: self))
//            .url(forResource: "deconv_expected_output", withExtension: "npy", subdirectory: "testdata")!
//        let expImg = MPSImage.loadFromNumpy(expUrl, destinationPixelFormat: testTextureFormatRGBA)
//        
//        XCTAssertEqual(outputImg, expImg)
//    }
    
    func testInterpixelStride() {
        let testImg1 = device.makeMPSImage(width: 2,
                                           height: 2,
                                           values: [[1,4,3,2],
                                                    [2,3,4,1],
                                                    [3,2,2,3],
                                                    [4,1,1,4]])

        
        let outputImg = device.makeMPSImage(width: 4,
                                            height: 4,
                                            values: [[0,0,0,0,
                                                      0,0,0,0,
                                                      0,0,0,0,
                                                      0,0,0,0],
                                                     [0,0,0,0,
                                                      0,0,0,0,
                                                      0,0,0,0,
                                                      0,0,0,0],
                                                     [0,0,0,0,
                                                      0,0,0,0,
                                                      0,0,0,0,
                                                      0,0,0,0],
                                                     [0,0,0,0,
                                                      0,0,0,0,
                                                      0,0,0,0,
                                                      0,0,0,0]])
        
        var s = UInt(2)
        let interpixelStride = ShaderRegistry.getDevice().makeBuffer(
            bytes: &s,
            length: MemoryLayout<UInt>.size,
            options: MTLResourceOptions.cpuCacheModeWriteCombined)
        
        /* Create our CommandEncoder*/
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.setComputePipelineState(ShaderRegistry.getOrLoad(name: "deconvolution_interpixel_stride"))
        encoder.setTexture(testImg1.texture, at: 0)
        encoder.setTexture(outputImg.texture, at: 1)
        encoder.setBuffer(interpixelStride, offset: 0, at: 2)
        let threadsPerGroup = MTLSizeMake(1, 1, 1)
        let threadGroups = MTLSizeMake(outputImg.texture.width,
                                       outputImg.texture.height, 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        /* Run our test */
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let expImg = device.makeMPSImage(width: 4,
                                         height: 4,
                                         values: [[1,0,4,0,
                                                   0,0,0,0,
                                                   3,0,2,0,
                                                   0,0,0,0],
                                                  [2,0,3,0,
                                                   0,0,0,0,
                                                   4,0,1,0,
                                                   0,0,0,0],
                                                  [3,0,2,0,
                                                   0,0,0,0,
                                                   2,0,3,0,
                                                   0,0,0,0],
                                                  [4,0,1,0,
                                                   0,0,0,0,
                                                   1,0,4,0,
                                                   0,0,0,0]])
        
        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }
    
    func testDeconvPart2Shader() {
        let bytes : [Float32] =
            [ -1,   0,   1,   2,  -2,   0,   2,   4,  -3,   0,   3,   6,  -4,   0,   4,
              8,  -5,   0,   5,  10,  -6,   0,   6,  12,  -7,   0,   7,  14,  -8,   0,
              8,  16,  -9,   0,   9,  18, -10,   0,  10,  20, -11,   0,  11,  22, -12,
              0,  12,  24, -13,   0,  13,  26, -14,   0,  14,  28, -15,   0,  15,  30,
              -16,   0,  16,  32]
        
        let inputBuffer = device.makeBuffer(bytes: bytes, length: bytes.count, options: MTLResourceOptions.storageModeShared)
        
        let outputImg = device.makeMPSImage(width: 5,
                                            height: 5,
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
            UInt32(64), // inputSize
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
    }
    
    
    //         func testIdentityNoPadding() {
    //             let testImg = device.MakeMPSImage(width: 2, height: 2, values: [1, 0,
    //                                                                             0, 1] as [Float32])
    //    
    //             /* Create our CommandEncoder */
    //             let w_pb = MemoryParameterBuffer([0, 0, 0,
    //                                               0, 1, 0,
    //                                               0, 0, 0])
    //             let b_pb = MemoryParameterBuffer(0)
    //             let deconv = DeconvolutionBlock(
    //                 kernelSize: 3,
    //                 channelsIn: 1,
    //                 channelsOut: 1,
    //                 w: w_pb,
    //                 b: b_pb,
    //                 relu: false,
    //                 padding: 0)
    //    
    //             let expImg = device.MakeMPSImage(width: 4,
    //                                              height: 4,
    //                                              pixelFormat: testTextureFormatR,
    //                                              values: [0, 0, 0, 0,
    //                                                       0, 1, 0, 0,
    //                                                       0, 0, 1, 0,
    //                                                       0, 0, 0, 0] as [Float32])
    //    
    //             /* Run our test */
    //             let outputImg = deconv.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
    //             execute()
    //    
    //             /* Verify the result */
    //             XCTAssertEqual(outputImg, expImg)
    //         }
    //    
    //         func testIdentityHalfPadding() {
    //             let testImg = device.MakeMPSImage(width: 4,
    //                                               height: 4,
    //                                               values: [0, 0, 0, 0,
    //                                                        0, 1, 0, 0,
    //                                                        0, 0, 1, 0,
    //                                                        0, 0, 0, 0] as [Float32])
    //    
    //             /* Create our CommandEncoder */
    //             let w_pb = MemoryParameterBuffer([0, 0, 0,
    //                                               0, 1, 0,
    //                                               0, 0, 0])
    //             let b_pb = MemoryParameterBuffer(0)
    //             let deconv = DeconvolutionBlock(
    //                 kernelSize: 3,
    //                 channelsIn: 1,
    //                 channelsOut: 1,
    //                 w: w_pb,
    //                 b: b_pb,
    //                 relu: false,
    //                 padding: 1)
    //    
    //             let expImg = device.MakeMPSImage(width: 4,
    //                                              height: 4,
    //                                              pixelFormat: testTextureFormatR,
    //                                              values: [0, 0, 0, 0,
    //                                                       0, 1, 0, 0,
    //                                                       0, 0, 1, 0,
    //                                                       0, 0, 0, 0] as [Float32])
    //    
    //             /* Run our test */
    //             let outputImg = deconv.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
    //             execute()
    //    
    //             /* Verify the result */
    //             XCTAssertEqual(outputImg, expImg)
    //         }
    //    
    //         func testIdentityFullPadding() {
    //             let testImg = device.MakeMPSImage(width: 6,
    //                                               height: 6,
    //                                               values: [0, 0, 0, 0, 0, 0,
    //                                                        0, 0, 0, 0, 0, 0,
    //                                                        0, 0, 1, 0, 1, 0,
    //                                                        0, 0, 0, 1, 0, 0,
    //                                                        0, 0, 0, 0, 0, 0,
    //                                                        0, 0, 0, 0, 0, 0] as [Float32])
    //             /* Create our CommandEncoder */
    //             let w_pb = MemoryParameterBuffer([0, 0, 0,
    //                                               0, 1, 0,
    //                                               0, 0, 0])
    //             let b_pb = MemoryParameterBuffer(0)
    //             let deconv = DeconvolutionBlock(
    //                 kernelSize: 3,
    //                 channelsIn: 1,
    //                 channelsOut: 1,
    //                 w: w_pb,
    //                 b: b_pb,
    //                 relu: false,
    //                 padding: 2)
    //    
    //             let expImg = device.MakeMPSImage(width: 4,
    //                                              height: 4,
    //                                              pixelFormat: testTextureFormatR,
    //                                              values: [0, 0, 0, 0,
    //                                                       0, 1, 0, 1,
    //                                                       0, 0, 1, 0,
    //                                                       0, 0, 0, 0] as [Float32])
    //    
    //             /* Run our test */
    //             let outputImg = deconv.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
    //             execute()
    //    
    //             /* Verify the result */
    //             XCTAssertEqual(outputImg, expImg)
    //         }
    //    
    //         func testSumHalfPaddingNonUnitStride() {
    //             let testImg = device.MakeMPSImage(width: 4,
    //                                               height: 4,
    //                                               values: [0, 0, 0, 0,
    //                                                        0, 1, 0, 0,
    //                                                        0, 0, 1, 0,
    //                                                        0, 0, 0, 0] as [Float32])
    //    
    //             /* Create our CommandEncoder */
    //             let w_pb = MemoryParameterBuffer([1, 1, 1,
    //                                               1, 1, 1,
    //                                               1, 1, 1])
    //             let b_pb = MemoryParameterBuffer(0)
    //             let deconv = DeconvolutionBlock(
    //                 kernelSize: 3,
    //                 channelsIn: 1,
    //                 channelsOut: 1,
    //                 w: w_pb,
    //                 b: b_pb,
    //                 relu: false,
    //                 padding: 1,
    //                 stride: 2)
    //    
    //             let expImg = device.MakeMPSImage(width: 7,
    //                                              height: 7,
    //                                              pixelFormat: testTextureFormatR,
    //                                              values: [0, 0, 0, 0, 0, 0, 0,
    //                                                       0, 1, 1, 1, 0, 0, 0,
    //                                                       0, 1, 1, 1, 0, 0, 0,
    //                                                       0, 1, 1, 2, 1, 1, 0,
    //                                                       0, 0, 0, 1, 1, 1, 0,
    //                                                       0, 0, 0, 1, 1, 1, 0,
    //                                                       0, 0, 0, 0, 0, 0, 0] as [Float32])
    //             
    //             /* Run our test */
    //             let outputImg = deconv.chain(MPSImageVariable(testImg)).forward(commandBuffer: commandBuffer)
    //             execute()
    //             
    //             /* Verify the result */
    //             XCTAssertEqual(outputImg, expImg)
    //         }
}
