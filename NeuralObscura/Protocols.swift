//
//  CommandEncoder.swift
//  NeuralObscura
//
//  Created by Edward Knox on 9/29/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

let textureFormat = MPSImageFeatureChannelFormat.float16

enum CommandEncoderError: Error {
    case chainMisconfiguration
}

protocol Chain {
    mutating func chain(_ top: CommandEncoder) -> CommandEncoder
}

protocol CommandEncoderDelegate {

    func getDestinationImageDescriptor(sourceImage: MPSImage?) -> MPSImageDescriptor
    
    func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage)
    
}

extension CommandEncoderDelegate {
    func identity(device: MTLDevice, commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage) throws {
//        let library = device.newDefaultLibrary()!
//        let identity = library.makeFunction(name: "identity")
//        let pipelineRGB = try device.makeComputePipelineState(function: identity!)
//        let pipelineBGR = try device.makeComputePipelineState(function: identity!)
//        
//        let identity = library.makeFunction(name: "identity")
//        pipelineIdentity = try device.makeComputePipelineState(function: identity!)
//        let encoder = commandBuffer.makeComputeCommandEncoder()
//        //encoder.setComputePipelineState(true ? pipelineBGR : pipelineRGB)
//        //encoder.setComputePipelineState(pipelineRGB)
//        //encoder.setComputePipelineState(pipelineBGR)
//        encoder.setComputePipelineState(pipelineIdentity)
//        encoder.setTexture(sourceTexture, at: 0)
//        //encoder.setTexture(adjustedMeanImage.texture, at: 1)
//        encoder.setTexture(outputImage.texture, at: 1)
//        let threadsPerGroups = MTLSizeMake(8, 8, 1)
//        /*            let threadGroups = MTLSizeMake(sourceTexture!.width / threadsPerGroups.width,
//         adjustedMeanImage.texture.height / threadsPerGroups.height, 1)*/
//        let threadGroups = MTLSizeMake(sourceTexture!.width / threadsPerGroups.width,
//                                       outputImage.texture.height / threadsPerGroups.height, 1)
//        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroups)
//        encoder.endEncoding()
//        adjustedMeanImage.readCount -= 1
//
    }
}

