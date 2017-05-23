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
    private var commandQueue: MTLCommandQueue!
    private var model: NeuralStyleModel!
    private var debugImagePaths: [String] = []
    private var debugImagePathIndex: Int = 0

    override func viewDidLoad() {
        super.viewDidLoad()
        commandQueue = ShaderRegistry.getDevice().makeCommandQueue()
        model = NeuralStyleModel(modelName: "composition", debug: true)

        debugImagePaths = [
            Bundle.main.path(forResource: "debug", ofType: "png")!,
            Bundle.main.path(forResource: "tubingen", ofType: "jpg")!]

        imageView.isUserInteractionEnabled = true
        loadNextDebugImage()
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

    @IBAction func imageViewTapDetected(_ sender: AnyObject) {
        loadNextDebugImage()
    }

    private func loadNextDebugImage() {
        let debugImagePath = debugImagePaths[debugImagePathIndex]
        imageView.image = UIImage.init(contentsOfFile: debugImagePath)!
        debugImagePathIndex += 1
        if (debugImagePathIndex >= debugImagePaths.count) {
            debugImagePathIndex = 0
        }
    }

    @IBAction func doStyling(_ sender: AnyObject) {
        let input = imageView.image!.toMPSImage(device: ShaderRegistry.getDevice())
        let output = model.execute(commandQueue: commandQueue, sourceImage: input)
        imageView.image! = output.toUIImage()
        print("done")
    }
}

