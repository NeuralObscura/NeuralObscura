//
//  SummationLayer.swift
//  NeuralObscura
//
//  Created by Edward Knox on 11/14/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class SummationLayer: BinaryCommandEncoder {
    init(debug: Bool = false) {
        super.init(
            delegate: SummationLayerDelegate(),
            debug: debug)
    }
}

class SummationLayerDelegate: CommandEncoderDelegate {
    private var sourceImageA: MPSImage!
    private var sourceImageB: MPSImage!
    private var inputsSupplied = 0
    
    init() {}
    
    func getDestinationImageDescriptor(sourceImage: MPSImage) -> MPSImageDescriptor {
        return MPSImageDescriptor(
            channelFormat: textureFormat,
            width: sourceImage.width,
            height: sourceImage.height,
            featureChannels: sourceImage.featureChannels)
    }
    
    
    func supplyInput(sourceImage: MPSImage, sourcePosition: Int) -> Bool {
        switch sourcePosition {
        case 0:
            self.sourceImageA = sourceImage
        case 1:
            self.sourceImageB = sourceImage
        default:
            fatalError("Invalid sourcePosition: \(sourcePosition)")
        }
        inputsSupplied += 1
        return inputsSupplied == 2
    }
    
    func encode(commandBuffer: MTLCommandBuffer, destinationImage: MPSImage) {
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.setComputePipelineState(ShaderRegistry.getOrLoad(name: "add"))
        encoder.setTexture(sourceImageA.texture, at: 0)
        encoder.setTexture(sourceImageB.texture, at: 1)
        encoder.setTexture(destinationImage.texture, at: 2)
        let threadsPerGroup = MTLSizeMake(1, 1, 1)
        let threadGroups = MTLSizeMake(destinationImage.texture.width,
                                       destinationImage.texture.height,
                                       destinationImage.texture.arrayLength)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        
        if let image = sourceImageA as? MPSTemporaryImage {
            image.readCount -= 1
        }
        if let image = sourceImageB as? MPSTemporaryImage {
            image.readCount -= 1
        }
        inputsSupplied = 0
    }
}
