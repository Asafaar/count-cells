# Count-cells-tool 🔬

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![WEKA](https://img.shields.io/badge/WEKA-Framework-orange.svg)](https://www.cs.waikato.ac.nz/ml/weka/)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)](https://github.com/yourusername/count-cells-tool)

## 📋 Overview

Count-cells-tool is an advanced machine learning solution built on the WEKA framework for automated cell counting and analysis in laboratory environments. Developed in collaboration with Ariel University's Department of Chemical Engineering, this tool revolutionizes cell culture research by automating crucial analysis processes.

### 🎯 Key Innovation

Our breakthrough achievement lies in developing a neural network model that achieves over 90% accuracy in cell classification using training data from a single cell image. This innovative approach enables highly customized models while significantly reducing development time and costs.

## ⭐ Features

### 🔋 Core Capabilities
- 🔍 Automated cell counting within plaques
- 📊 Live/dead cell ratio analysis
- ⏰ Optimal cell splitting time prediction
- 🦠 Early infection detection

### 📈 Benefits
- 📉 Reduced cell loss through precise splitting timing
- 🚨 Early detection of culture infections
- 👨‍🔬 Significantly reduced manual workload for researchers
- ✅ Improved accuracy in cell counting and assessment

## 🚀 Installation

### 👥 For Users

run the executable file in the dist folder

### 👨‍💻 For Developers

Follow these steps to set up the development environment:

1. Install Fiji image processing software

2. Create Trainable Weka Segmentation:
   - Required inputs:
     - Original image
     - "Create result" picture

3. Organize Images:
   ```
   src/
   ├── image-to-txt-images/
   │   ├── weka/
   │   │   └── segmentation_images
   │   └── original/
   │       └── original_images
   ```

4. Generate YOLO Format Data:
```bash
python from-seg-to-txt-for-yolo.py
```

5. Prepare Dataset:
   ```
   dataset/
   ├── labels/
   │   └── label_files
   └── images/
       └── original_images
   ```

6. Train the Model:
```bash
python modeltrain/trainmodel.py
```
The trained model will be saved in the `runs` folder under your latest training session.

7. Run Prediction Interface:
```bash
python gui-predict.py
```

## 📸 Visual Examples

### Cell Analysis Process
<img src="https://github.com/Asafaar/count-cells/blob/Work/README-pic/weka-seg.png" width="500" height="400">
<img src="https://github.com/Asafaar/count-cells/blob/Work/README-pic/class%20image.png" width="500" height="400">

### Results Visualization

<img src="https://github.com/Asafaar/count-cells/blob/Work/README-pic/gui.png" width="500" >
<img src="https://github.com/Asafaar/count-cells/blob/Work/README-pic/predict-results.png" width="500" >

## 🔮 Future Development

Plans for future enhancements include:
- 🔄 Integration with laboratory incubators for automatic scanning
- ⚡ Real-time analysis capabilities
- 🤖 Enhanced automation features for continuous monitoring

## 📊 Project Status

Current Version: 1.0.0 beta

## 📫 Contact

For questions and support, please reach out to:
- 📧 Email: asaf9512@gmail.com

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
