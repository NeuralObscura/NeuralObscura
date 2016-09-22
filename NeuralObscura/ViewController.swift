//
//  ViewController.swift
//  NeuralObscura
//
//  Created by Paul Bergeron on 9/15/16.
//  Copyright © 2016 Paul Bergeron. All rights reserved.
//

import UIKit
import MetalKit
import MetalPerformanceShaders

class ViewController: UIViewController, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
    @IBOutlet weak var imageView: UIImageView!
    private var device: MTLDevice!
    private var ciContext : CIContext!
    private var textureLoader : MTKTextureLoader!
    private var commandQueue: MTLCommandQueue!
    var sourceTexture : MTLTexture? = nil

    override func viewDidLoad() {
        super.viewDidLoad()

        device = MTLCreateSystemDefaultDevice()

        guard MPSSupportsMTLDevice(device) else {
            print("Error: Metal Performance Shaders not supported on this device")
            return
        }

        ciContext = CIContext.init(mtlDevice: device)

        textureLoader = MTKTextureLoader(device: device!)

        commandQueue = device!.makeCommandQueue()

    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }

    @IBAction func takePicture(_ sender: AnyObject) {
        let imagePicker = UIImagePickerController()
        // If the device has a camera, take a picture, otherwise,
        // just pick from photo library
        if UIImagePickerController.isSourceTypeAvailable(.camera) {
            imagePicker.sourceType = .camera
        } else {
            imagePicker.sourceType = .photoLibrary
        }

        imagePicker.delegate = self

        present(imagePicker, animated: true, completion: nil)
    }

    @IBAction func takePictureFromLibrary(_ sender: AnyObject) {
        let imagePicker = UIImagePickerController()
        imagePicker.sourceType = .photoLibrary

        imagePicker.delegate = self

        present(imagePicker, animated: true, completion: nil)
    }

    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        let image = info[UIImagePickerControllerOriginalImage] as! UIImage
        imageView.image = image

        dismiss(animated: true, completion: nil)
    }

    @IBAction func doStyling(_ sender: AnyObject) {
        // The custom compute kernels for preprocessing the input images.
        let pipelineRGB: MTLComputePipelineState
        let pipelineBGR: MTLComputePipelineState
        let outputImage: MPSImage

        sourceTexture = imageView.image!.createMTLTextureForDevice(device: self.device)

        // Before we pass an image into the network, we need to adjust its RGB
        // values. This is done with a custom compute kernel. Here we load that
        // kernel (from Shaders.metal) and set up the compute pipeline.
        do {
            let library = device.newDefaultLibrary()!
            let adjust_mean_rgb = library.makeFunction(name: "adjust_mean_rgb")
            pipelineRGB = try device.makeComputePipelineState(function: adjust_mean_rgb!)

            let adjust_mean_bgr = library.makeFunction(name: "adjust_mean_bgr")
            pipelineBGR = try device.makeComputePipelineState(function: adjust_mean_bgr!)
        } catch {
            fatalError("Error initializing compute pipeline")
        }

        let input_id  = MPSImageDescriptor(channelFormat: .float16, width: 224, height: 224, featureChannels: 3)

        autoreleasepool{
            let commandBuffer = commandQueue.makeCommandBuffer()

            let adjustedMeanImage = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: input_id)

            let encoder = commandBuffer.makeComputeCommandEncoder()
            //encoder.setComputePipelineState(true ? pipelineBGR : pipelineRGB)
            encoder.setComputePipelineState(pipelineRGB)
            //encoder.setComputePipelineState(pipelineBGR)
            encoder.setTexture(sourceTexture, at: 0)
            encoder.setTexture(adjustedMeanImage.texture, at: 1)
            let threadsPerGroups = MTLSizeMake(8, 8, 1)
            let threadGroups = MTLSizeMake(sourceTexture!.width / threadsPerGroups.width,
                                           adjustedMeanImage.texture.height / threadsPerGroups.height, 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroups)
            encoder.endEncoding()
            adjustedMeanImage.readCount -= 1

            // Tell the GPU to start and wait until it's done.
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            //TODO: get output image by running model
            //let model = CompositionModel()
        }

        print("done")

        if(Int(imageView.image!.size.width) == sourceTexture!.width) {
            imageView.image = UIImage.MTLTextureToUIImage(texture: sourceTexture!, orientation: UIImageOrientation.up)
        } else {
            imageView.image = UIImage.MTLTextureToUIImage(texture: sourceTexture!, orientation: UIImageOrientation.right)
        }
    }
}

