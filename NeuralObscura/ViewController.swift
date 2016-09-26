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
        let pipelineIdentity: MTLComputePipelineState
        let outputImage: MPSImage

        sourceTexture = imageView.image!.createMTLTextureForDevice(device: self.device)

        let output_id = MPSImageDescriptor(channelFormat: .unorm8, width: sourceTexture!.width, height: sourceTexture!.height, featureChannels: 3)
        outputImage = MPSImage(device: device, imageDescriptor: output_id)

        do {
            let library = device.newDefaultLibrary()!
            let identity = library.makeFunction(name: "identity")
            pipelineIdentity = try device.makeComputePipelineState(function: identity!)
        } catch {
            fatalError("Error initializing compute pipeline")
        }

        let model = CompositionModel(device: device)

        autoreleasepool {
            let commandBuffer = commandQueue.makeCommandBuffer()

            let encoder = commandBuffer.makeComputeCommandEncoder()

            encoder.setComputePipelineState(pipelineIdentity)
            encoder.setTexture(sourceTexture, at: 0)
            encoder.setTexture(outputImage.texture, at: 1)
            /* Instructions for optimizing thread configuration here:
                https://developer.apple.com/library/prerelease/content/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Compute-Ctx/Compute-Ctx.html#//apple_ref/doc/uid/TP40014221-CH6-SW2
            */
            let threadsPerGroups = MTLSizeMake(8, 8, 1)
            let threadGroups = MTLSizeMake(sourceTexture!.width / threadsPerGroups.width,
                                           outputImage.texture.height / threadsPerGroups.height, 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroups)
            encoder.endEncoding()

            // Tell the GPU to start and wait until it's done.
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            //TODO: get output image by running model

        }

        print("done")

        //setImageViewToTexture(texture: sourceTexture!)
        //sleep(3)
        setImageViewToTexture(texture: outputImage.texture)
    }

    func setImageViewToTexture(texture: MTLTexture) {
        if(Int(imageView.image!.size.width) == texture.width) {
            imageView.image = UIImage.MTLTextureToUIImage(texture: texture, orientation: UIImageOrientation.up)
        } else {
            imageView.image = UIImage.MTLTextureToUIImage(texture: texture, orientation: UIImageOrientation.right)
        }
    }
}

