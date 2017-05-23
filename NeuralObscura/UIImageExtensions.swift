//
// Created by Edward Knox on 5/20/17.
// Copyright (c) 2017 Paul Bergeron. All rights reserved.
//

import UIKit
import Foundation
import Metal
import MetalKit
import MetalPerformanceShaders

extension UIImage {

    func toMPSImage(device: MTLDevice) -> MPSImage {
        return MPSImage(texture: self.toMTLTexture(device: device, debug: false), featureChannels: 3)
    }

    func toMTLTexture(device: MTLDevice) -> MTLTexture {
        return self.toMTLTexture(device: device, debug: true)
    }

    func toMTLTexture(device: MTLDevice, debug: Bool) -> MTLTexture {
        let ciContext = CIContext.init(mtlDevice: device)
        var options: [String : NSObject] = [
            MTKTextureLoaderOptionTextureUsage:         MTLTextureUsage.shaderRead.rawValue as NSObject,
            MTKTextureLoaderOptionOrigin:               MTKTextureLoaderOriginTopLeft as NSObject,
        ]

        if (debug) {
            options[MTKTextureLoaderOptionTextureStorageMode] = MTLStorageMode.shared.rawValue as NSObject
        } else {
            options[MTKTextureLoaderOptionTextureStorageMode] = MTLStorageMode.private.rawValue as NSObject
        }

        let cgImage: CGImage
        if let img = self.cgImage {
            cgImage = img
        } else {
            let ciImage = CIImage(image: self)
            cgImage = ciContext.createCGImage(ciImage!, from: ciImage!.extent)!
        }

        let textureLoader = MTKTextureLoader(device: device)
        return try! textureLoader.newTexture(with: cgImage, options: options)
    }

}
