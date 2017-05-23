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
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let ciContext = CIContext.init(mtlDevice: device)

        let cgImage: CGImage
        if let img = self.cgImage {
            cgImage = img
        } else {
            let ciImage = CIImage(image: self)
            cgImage = ciContext.createCGImage(ciImage!, from: ciImage!.extent)!
        }
        let textureLoader = MTKTextureLoader(device: device)
        let texture = try! textureLoader.newTexture(with: cgImage)
        return MPSImage(texture: texture, featureChannels: 3)
    }

}
