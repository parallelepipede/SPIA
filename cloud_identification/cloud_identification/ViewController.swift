//
//  ViewController.swift
//  cloud_identification
//
//  Created by Zanoncelli on 24/07/2021.
//  Copyright © 2021 Zanoncelli. All rights reserved.
//

import UIKit

class ViewController: UIViewController {

    @IBOutlet var imageView: UIImageView!
    @IBOutlet var button: UIButton!
    override func viewDidLoad() {
        super.viewDidLoad()
        
        imageView.backgroundColor = .gray
        
        button.backgroundColor = .blue
        button.setTitle("Take Cloud Picture", for: .normal)
        button.setTitleColor(.white, for: .normal)
        
    }
    @IBAction func didTapButton(){
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = self
        present(picker,animated: true)
        
    }
}

extension ViewController : UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.dismiss(animated: true, completion: nil)
    }
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        picker.dismiss(animated: true, completion: nil)
        guard let image = info[UIImagePickerController.InfoKey.originalImage] as? UIImage else {
            return
        } // cause it might be nil
        imageView.image = image
    }
}

