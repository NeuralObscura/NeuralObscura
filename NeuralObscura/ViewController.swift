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
    private var model: NeuralStyleModel!

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
        
        // This is computationally expensive, should optimize
        // by initializing on a background thread.
        model = NeuralStyleModel(device: device, modelName: "composition")

        let debugImagePath = Bundle.main.path(forResource: "tubingen", ofType: "jpg")!

        imageView.image = UIImage.init(contentsOfFile: debugImagePath)!
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

    private func image(from texture: MTLTexture) -> MPSImage {
        // We set featureChannels to 3 because the neural network is only trained
        // on RGB data (the first 3 channels), not alpha (the 4th channel).
        return MPSImage(texture: texture, featureChannels: 3)
    }

    @IBAction func doStyling(_ sender: AnyObject) {
        // note the configurable options
        let input = imageView.image!
        let inputMtlTexture = input.createMTLTextureForDevice(device: device)
        let output = model.forward(commandQueue: commandQueue, sourceImage: image(from: inputMtlTexture))
        print("-------")
        imageView.image!.fourCorners(device: self.device)
        print("-------")
        print("done")
        //TODO: fix me
        /*
        if(Int(input.size.width) == inputMtlTexture.width) {
            imageView.image! = UIImage.MTLTextureToUIImage(texture: outputMtlTexture, orientation: UIImageOrientation.up)
        } else {
            imageView.image! = UIImage.MTLTextureToUIImage(texture: outputMtlTexture, orientation: UIImageOrientation.right)
        }
 */
    }
}

