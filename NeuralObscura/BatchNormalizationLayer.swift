//
//  BatchNormalizationLayer.swift
//  NeuralObscura
//
//  Created by Edward Knox on 9/27/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class BatchNormalizationLayer: CommandEncoder {
    init(
        device: MTLDevice,
        channelsIn: Int,
        beta: UnsafePointer<Float>,
        gamma: UnsafePointer<Float>,
        useTemporary: Bool = true) {
        super.init(
            device: device,
            delegate: BatchNormalizationLayerDelegate(
                device: device,
                channelsIn: channelsIn,
                beta: beta,
                gamma: gamma),
            useTemporary: useTemporary)
    }
}

class BatchNormalizationLayerDelegate: CommandEncoderDelegate {
    let beta: UnsafePointer<Float>
    let gamma: UnsafePointer<Float>
    let channelsIn: Int
    
    init(device: MTLDevice, channelsIn: Int, beta: UnsafePointer<Float>, gamma: UnsafePointer<Float>) {
        self.channelsIn = Int(channelsIn)
        self.beta = beta
        self.gamma = gamma
    }
    
    func getDestinationImageDescriptor(sourceImage: MPSImage) -> MPSImageDescriptor {
        return MPSImageDescriptor(
            channelFormat: textureFormat,
            width: sourceImage.width,
            height: sourceImage.height,
            featureChannels: sourceImage.featureChannels)
    }
    
    func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage) {
        print("bn encode")
        // TODO: encode batch normalization
        if sourceImage is MPSTemporaryImage {
            (sourceImage as! MPSTemporaryImage).readCount -= 1
        }
    }
}
