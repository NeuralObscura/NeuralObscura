//
//  ShaderTests.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 10/31/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import XCTest
import MetalKit
import MetalPerformanceShaders
@testable import NeuralObscura

class ShaderTests: CommandEncoderBaseTest {

    func testIdentity() {
        let testImg = device.MakeTestMPSImage(width: 4, height: 4, values: [0, 0, 0, 0,
                                                                            0, 1, 0, 1,
                                                                            0, 0, 1, 0,
                                                                            0, 0, 0, 0])

        let outputImg = device.MakeTestMPSImage(width: 4, height: 4, values: [0, 0, 0, 0,
                                                                              0, 0, 0, 0,
                                                                              0, 0, 0, 0,
                                                                              0, 0, 0, 0])
        var identity: MTLComputePipelineState
        do {
            let library = device.newDefaultLibrary()!
            let identityFunc = library.makeFunction(name: "identity")
            identity = try device.makeComputePipelineState(function: identityFunc!)
        } catch {
            fatalError("Error initializing compute pipeline")
        }

        print(testImg)

        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.setComputePipelineState(identity)
        encoder.setTexture(testImg.texture, at: 0)
        encoder.setTexture(outputImg.texture, at: 1)
        let threadsPerGroup = MTLSizeMake(1, 1, 1)
        let threadGroups = MTLSizeMake(outputImg.texture.width / threadsPerGroup.width,
                                       outputImg.texture.height / threadsPerGroup.height, 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        /* Run our test */
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        print(outputImg)


        let expImg = device.MakeTestMPSImage(width: 4, height: 4, values: [0, 0, 0, 0,
                                                                           0, 1, 0, 1,
                                                                           0, 0, 1, 0,
                                                                           0, 0, 0, 0])

        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }

    struct BufferTextureImage {
        var buffer: MTLBuffer
        var texture: MTLTexture
        var image: MPSImage
    }


    // NOT WORKING
    func makeBufferTextureImage(size: MTLSize, values: [[UInt8]]) -> BufferTextureImage {
        var reduced = values.reduce([], +)
        var buffer: MTLBuffer? = nil
        withUnsafePointer(to: &reduced) {
            buffer = device.makeBuffer(bytes: $0, length: reduced.count, options: MTLResourceOptions.storageModeShared)
        }

        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.textureType = MTLTextureType.type2DArray
        textureDescriptor.arrayLength = size.depth
        textureDescriptor.width = size.width
        textureDescriptor.height = size.height
        textureDescriptor.pixelFormat = .r8Unorm

        debugPrint(size.width * MemoryLayout<Float32>.size)

        // BROKEN; can't create an array of textures from a single buffer!
        let texture = buffer!.makeTexture(descriptor: textureDescriptor, offset: 0, bytesPerRow: size.width * MemoryLayout<Float32>.size)

        let image = MPSImage(texture: texture, featureChannels: size.depth)

        return BufferTextureImage(buffer: buffer!, texture: texture, image: image)
    }

/*    func testMakeBufferTextureImage() {
        let size = MTLSizeMake(2, 2, 5)
        let values: [[UInt8]] = [[1,1,
                                  0,0],
                                 [1,1,
                                  0,0],
                                 [1,1,
                                  0,0],
                                 [1,1,
                                  0,0],
                                 [1,1,
                                  0,0]]
        let btl = makeBufferTextureImage(size: size, values: values)

        print(btl.image)


        XCTAssertEqual(1,0)
    }*/

    func testAddOne() {
        /*
        let testImg = device.MakeDeepTestMPSImage(width: 2, height: 2, values: [[1,1,1,1,1,1,1,1,
                                                                                 0,0,0,0,0,0,0,0],
                                                                                [1,1,1,1,1,1,1,1,
                                                                                 0,0,0,0,0,0,0,0],
                                                                                [1,1,1,1,1,1,1,1,
                                                                                 0,0,0,0,0,0,0,0],
                                                                                [1,1,1,1,1,1,1,1,
                                                                                 0,0,0,0,0,0,0,0],
                                                                                [1,1,1,1,1,1,1,1,
                                                                                 0,0,0,0,0,0,0,0]])

        let outputImg = device.MakeDeepTestMPSImage(width: 2, height: 2, values: [[0,0,0,0,0,0,0,0,
                                                                                   0,0,0,0,0,0,0,0],
                                                                                  [0,0,0,0,0,0,0,0,
                                                                                   0,0,0,0,0,0,0,0],
                                                                                  [0,0,0,0,0,0,0,0,
                                                                                   0,0,0,0,0,0,0,0],
                                                                                  [0,0,0,0,0,0,0,0,
                                                                                   0,0,0,0,0,0,0,0],
                                                                                  [0,0,0,0,0,0,0,0,
                                                                                   0,0,0,0,0,0,0,0]])
 */
        let testImg = device.MakeDeepTestMPSImage(width: 2, height: 2, values: [[1,1,
                                                                                 0,0]])

        let outputImg = device.MakeDeepTestMPSImage(width: 2, height: 2, values: [[0,0,
                                                                                   0,0]])

        var addOne: MTLComputePipelineState
        do {
            let library = device.newDefaultLibrary()!
            let addOneFunc = library.makeFunction(name: "addOne")
            addOne = try device.makeComputePipelineState(function: addOneFunc!)
        } catch {
            fatalError("Error initializing compute pipeline")
        }

        print(testImg)

        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.setComputePipelineState(addOne)
        encoder.setTexture(testImg.texture, at: 0)
        encoder.setTexture(outputImg.texture, at: 1)
        let threadsPerGroups = MTLSizeMake(1, 1, 1)
        let threadGroups = MTLSizeMake(outputImg.texture.width,
                                       outputImg.texture.height, outputImg.texture.arrayLength)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroups)
        encoder.endEncoding()
        /* Run our test */
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        print(outputImg)


        let expImg = device.MakeDeepTestMPSImage(width: 2, height: 2, values: [[1,1,
                                                                                0,0]])

        /* Verify the result */
        XCTAssertEqual(outputImg, expImg)
    }

}
