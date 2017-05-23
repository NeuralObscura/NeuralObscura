//
//  Globals.swift
//  NeuralObscura
//
//  Created by Edward Knox on 11/13/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

let textureFormat = MPSImageFeatureChannelFormat.float16

var testTextureFormatRGBA: MTLPixelFormat {
    switch textureFormat {
    case MPSImageFeatureChannelFormat.float16:
        return MTLPixelFormat.rgba16Float
    default:
        return MTLPixelFormat.rgba32Float
    }
}

var testTextureFormatR: MTLPixelFormat {
    switch textureFormat {
    case MPSImageFeatureChannelFormat.float16:
        return MTLPixelFormat.r16Float
    default:
        return MTLPixelFormat.r32Float
    }
}

let blockSize = 8192
let ExpectedFloat16Size = 2  /// Not MemoryLayout<UInt16>.size
