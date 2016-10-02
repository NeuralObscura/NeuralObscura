//
//  NeuralStyleModel.swift
//  NeuralObscura
//
//  Created by Edward Knox on 10/2/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class NeuralStyleModel {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let modelName: String
    private let useTemporary: Bool
    
    private var modelParams = [String: StyleModelData]()
    
    private let inputVariable: InputVariable
    private let model: NeuralStyleBlock
    private let outputVariable: OutputVariable
    
    init(
        device: MTLDevice,
        commandQueue: MTLCommandQueue,
        modelName: String,
        useTemporary: Bool = true) {
        self.device = device
        self.commandQueue = commandQueue
        self.modelName = modelName
        self.useTemporary = useTemporary
        
        loadModelParams()
        
        // TODO: Implement InputVariable
        inputVariable = InputVariable(/* give some shape here */)
        model = NeuralStyleBlock(device: device, commandQueue: commandQueue, modelParams: modelParams)
        // TODO: Implement OutputVariable
        outputVariable = OutputVariable(/* No shape needed I don't think */)
        
        outputVariable.chain(model.chain(inputVariable))
    }
    
    func forward(sourceImage: CGImage) -> MPSImage {
        // TODO: Do image conversion -> MPSImage
        inputVariable.encode(/* output of conversion */)
    }
    
    func loadModelParams() {
        /* Load model parameters */
        modelParams["r4_c2_W"] = StyleModelData(modelName: modelName, rawFileName: "r4_c2_W")
        //r4_c2_W shape = (128, 128, 3, 3)
        modelParams["r4_c2_b"] = StyleModelData(modelName: modelName, rawFileName: "r4_c2_b")
        //r4_c2_b shape = (128,)
        modelParams["r4_c1_W"] = StyleModelData(modelName: modelName, rawFileName: "r4_c1_W")
        //r4_c1_W shape = (128, 128, 3, 3)
        modelParams["r4_c1_b"] = StyleModelData(modelName: modelName, rawFileName: "r4_c1_b")
        //r4_c1_b shape = (128,)
        modelParams["r4_b1_gamma"] = StyleModelData(modelName: modelName, rawFileName: "r4_b1_gamma")
        //r4_b1_gamma shape = (128,)
        modelParams["r4_b1_beta"] = StyleModelData(modelName: modelName, rawFileName: "r4_b1_beta")
        //r4_b1_beta shape = (128,)
        modelParams["r4_b2_gamma"] = StyleModelData(modelName: modelName, rawFileName: "r4_b2_gamma")
        //r4_b2_gamma shape = (128,)
        modelParams["r4_b2_beta"] = StyleModelData(modelName: modelName, rawFileName: "r4_b2_beta")
        //r4_b2_beta shape = (128,)
        modelParams["r5_c2_W"] = StyleModelData(modelName: modelName, rawFileName: "r5_c2_W")
        //r5_c2_W shape = (128, 128, 3, 3)
        modelParams["r5_c2_b"] = StyleModelData(modelName: modelName, rawFileName: "r5_c2_b")
        //r5_c2_b shape = (128,)
        modelParams["r5_c1_W"] = StyleModelData(modelName: modelName, rawFileName: "r5_c1_W")
        //r5_c1_W shape = (128, 128, 3, 3)
        modelParams["r5_c1_b"] = StyleModelData(modelName: modelName, rawFileName: "r5_c1_b")
        //r5_c1_b shape = (128,)
        modelParams["r5_b1_gamma"] = StyleModelData(modelName: modelName, rawFileName: "r5_b1_gamma")
        //r5_b1_gamma shape = (128,)
        modelParams["r5_b1_beta"] = StyleModelData(modelName: modelName, rawFileName: "r5_b1_beta")
        //r5_b1_beta shape = (128,)
        modelParams["r5_b2_gamma"] = StyleModelData(modelName: modelName, rawFileName: "r5_b2_gamma")
        //r5_b2_gamma shape = (128,)
        modelParams["r5_b2_beta"] = StyleModelData(modelName: modelName, rawFileName: "r5_b2_beta")
        //r5_b2_beta shape = (128,)
        modelParams["r1_c2_W"] = StyleModelData(modelName: modelName, rawFileName: "r1_c2_W")
        //r1_c2_W shape = (128, 128, 3, 3)
        modelParams["r1_c2_b"] = StyleModelData(modelName: modelName, rawFileName: "r1_c2_b")
        //r1_c2_b shape = (128,)
        modelParams["r1_c1_W"] = StyleModelData(modelName: modelName, rawFileName: "r1_c1_W")
        //r1_c1_W shape = (128, 128, 3, 3)
        modelParams["r1_c1_b"] = StyleModelData(modelName: modelName, rawFileName: "r1_c1_b")
        //r1_c1_b shape = (128,)
        modelParams["r1_b1_gamma"] = StyleModelData(modelName: modelName, rawFileName: "r1_b1_gamma")
        //r1_b1_gamma shape = (128,)
        modelParams["r1_b1_beta"] = StyleModelData(modelName: modelName, rawFileName: "r1_b1_beta")
        //r1_b1_beta shape = (128,)
        modelParams["r1_b2_gamma"] = StyleModelData(modelName: modelName, rawFileName: "r1_b2_gamma")
        //r1_b2_gamma shape = (128,)
        modelParams["r1_b2_beta"] = StyleModelData(modelName: modelName, rawFileName: "r1_b2_beta")
        //r1_b2_beta shape = (128,)
        modelParams["r2_c2_W"] = StyleModelData(modelName: modelName, rawFileName: "r2_c2_W")
        //r2_c2_W shape = (128, 128, 3, 3)
        modelParams["r2_c2_b"] = StyleModelData(modelName: modelName, rawFileName: "r2_c2_b")
        //r2_c2_b shape = (128,)
        modelParams["r2_c1_W"] = StyleModelData(modelName: modelName, rawFileName: "r2_c1_W")
        //r2_c1_W shape = (128, 128, 3, 3)
        modelParams["r2_c1_b"] = StyleModelData(modelName: modelName, rawFileName: "r2_c1_b")
        //r2_c1_b shape = (128,)
        modelParams["r2_b1_gamma"] = StyleModelData(modelName: modelName, rawFileName: "r2_b1_gamma")
        //r2_b1_gamma shape = (128,)
        modelParams["r2_b1_beta"] = StyleModelData(modelName: modelName, rawFileName: "r2_b1_beta")
        //r2_b1_beta shape = (128,)
        modelParams["r2_b2_gamma"] = StyleModelData(modelName: modelName, rawFileName: "r2_b2_gamma")
        //r2_b2_gamma shape = (128,)
        modelParams["r2_b2_beta"] = StyleModelData(modelName: modelName, rawFileName: "r2_b2_beta")
        //r2_b2_beta shape = (128,)
        modelParams["r3_c2_W"] = StyleModelData(modelName: modelName, rawFileName: "r3_c2_W")
        //r3_c2_W shape = (128, 128, 3, 3)
        modelParams["r3_c2_b"] = StyleModelData(modelName: modelName, rawFileName: "r3_c2_b")
        //r3_c2_b shape = (128,)
        modelParams["r3_c1_W"] = StyleModelData(modelName: modelName, rawFileName: "r3_c1_W")
        //r3_c1_W shape = (128, 128, 3, 3)
        modelParams["r3_c1_b"] = StyleModelData(modelName: modelName, rawFileName: "r3_c1_b")
        //r3_c1_b shape = (128,)
        modelParams["r3_b1_gamma"] = StyleModelData(modelName: modelName, rawFileName: "r3_b1_gamma")
        //r3_b1_gamma shape = (128,)
        modelParams["r3_b1_beta"] = StyleModelData(modelName: modelName, rawFileName: "r3_b1_beta")
        //r3_b1_beta shape = (128,)
        modelParams["r3_b2_gamma"] = StyleModelData(modelName: modelName, rawFileName: "r3_b2_gamma")
        //r3_b2_gamma shape = (128,)
        modelParams["r3_b2_beta"] = StyleModelData(modelName: modelName, rawFileName: "r3_b2_beta")
        //r3_b2_beta shape = (128,)
        modelParams["b4_gamma"] = StyleModelData(modelName: modelName, rawFileName: "b4_gamma")
        //b4_gamma shape = (64,)
        modelParams["b4_beta"] = StyleModelData(modelName: modelName, rawFileName: "b4_beta")
        //b4_beta shape = (64,)
        modelParams["b5_gamma"] = StyleModelData(modelName: modelName, rawFileName: "b5_gamma")
        //b5_gamma shape = (32,)
        modelParams["b5_beta"] = StyleModelData(modelName: modelName, rawFileName: "b5_beta")
        //b5_beta shape = (32,)
        modelParams["b1_gamma"] = StyleModelData(modelName: modelName, rawFileName: "b1_gamma")
        //b1_gamma shape = (32,)
        modelParams["b1_beta"] = StyleModelData(modelName: modelName, rawFileName: "b1_beta")
        //b1_beta shape = (32,)
        modelParams["b2_gamma"] = StyleModelData(modelName: modelName, rawFileName: "b2_gamma")
        //b2_gamma shape = (64,)
        modelParams["b2_beta"] = StyleModelData(modelName: modelName, rawFileName: "b2_beta")
        //b2_beta shape = (64,)
        modelParams["b3_gamma"] = StyleModelData(modelName: modelName, rawFileName: "b3_gamma")
        //b3_gamma shape = (128,)
        modelParams["b3_beta"] = StyleModelData(modelName: modelName, rawFileName: "b3_beta")
        //b3_beta shape = (128,)
        modelParams["c3_W"] = StyleModelData(modelName: modelName, rawFileName: "c3_W")
        //c3_W shape = (128, 64, 4, 4)
        modelParams["c3_b"] = StyleModelData(modelName: modelName, rawFileName: "c3_b")
        //c3_b shape = (128,)
        modelParams["c2_W"] = StyleModelData(modelName: modelName, rawFileName: "c2_W")
        //c2_W shape = (64, 32, 4, 4)
        modelParams["c2_b"] = StyleModelData(modelName: modelName, rawFileName: "c2_b")
        //c2_b shape = (64,)
        modelParams["c1_W"] = StyleModelData(modelName: modelName, rawFileName: "c1_W")
        //c1_W shape = (32, 3, 9, 9)
        modelParams["c1_b"] = StyleModelData(modelName: modelName, rawFileName: "c1_b")
        //c1_b shape = (32,)
        modelParams["d2_W"] = StyleModelData(modelName: modelName, rawFileName: "d2_W")
        //d2_W shape = (64, 32, 4, 4)
        modelParams["d2_b"] = StyleModelData(modelName: modelName, rawFileName: "d2_b")
        //d2_b shape = (32,)
        modelParams["d3_W"] = StyleModelData(modelName: modelName, rawFileName: "d3_W")
        //d3_W shape = (32, 3, 9, 9)
        modelParams["d3_b"] = StyleModelData(modelName: modelName, rawFileName: "d3_b")
        //d3_b shape = (3,)
        modelParams["d1_W"] = StyleModelData(modelName: modelName, rawFileName: "d1_W")
        //d1_W shape = (128, 64, 4, 4)
        modelParams["d1_b"] = StyleModelData(modelName: modelName, rawFileName: "d1_b")
        //d1_b shape = (64,)
    }
}
