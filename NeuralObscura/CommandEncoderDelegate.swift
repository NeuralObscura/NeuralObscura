//
//  CommandEncoderDelegate.swift
//  NeuralObscura
//
//  Created by Edward Knox on 11/13/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

protocol CommandEncoderDelegate {
    func getDestinationImageDescriptor(sourceImage: MPSImage) -> MPSImageDescriptor
    
    /* returns true if all inputs have been supplied and delegate is ready for encode() call */
    func supplyInput(sourceImage: MPSImage, sourcePosition: Int) -> Bool
    
    func encode(commandBuffer: MTLCommandBuffer, destinationImage: MPSImage)
}
