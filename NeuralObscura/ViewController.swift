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
    let bytesPerPixel: Int = 4

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

        var image = imageView.image!.cgImage
        if (image == nil) {
            let ciImage = CIImage(image: imageView.image!)
            image = ciContext.createCGImage(ciImage!, from: ciImage!.extent)
        }

        do {
            sourceTexture = try textureLoader.newTexture(with: image!, options: [:])
        }
        catch let error as NSError {
            fatalError("Unexpected error ocurred: \(error.localizedDescription).")
        }

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
            encoder.setComputePipelineState(true ? pipelineBGR : pipelineRGB)
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

        imageView.image = imageFromTexture(texture: sourceTexture!)
    }

    func imageFromTexture(texture: MTLTexture) -> UIImage {

        // The total number of bytes of the texture
        let imageByteCount = texture.width * texture.height * bytesPerPixel

        // The number of bytes for each image row
        let bytesPerRow = texture.width * bytesPerPixel

        // An empty buffer that will contain the image
        var src = [UInt8](repeating: 0, count: Int(imageByteCount))

        // Gets the bytes from the texture
        let region = MTLRegionMake2D(0, 0, texture.width, texture.height)
        texture.getBytes(&src, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)

        // Creates an image context
        let bitmapInfo = CGBitmapInfo(rawValue: (CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue))
        let bitsPerComponent = 8
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: &src, width: texture.width, height: texture.height, bitsPerComponent: bitsPerComponent, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: bitmapInfo.rawValue);

        // Creates the image from the graphics context
        let dstImage = context!.makeImage();

        // Creates the final UIImage
        return UIImage(cgImage: dstImage!, scale: 0.0, orientation: UIImageOrientation.downMirrored)
    }

}

