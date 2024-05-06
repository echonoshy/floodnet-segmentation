# FloodNet Dataset Image Semantic Segmentation Technical Report

## Introduction

This code and its associated content were developed to implement an image semantic segmentation task on the supervised FloodNet dataset. The goal is to design a simple framework that allows for quick model construction, training, prediction, and visualization.

### Experiment Environment
- MacOS 14.4
- Linux
  - PyTorch: 2.1.0
  - Python: 3.10 (Ubuntu 22.04)
  - CUDA: 12.1
  - GPU: RTX 4090 (24GB) * 1
  - CPU: 12 vCPU Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz

### Code Structure and Functionality
- `config`: Configuration files, containing model-related parameters and the currently selected model for training
- `core`: Main training code for the models
- `exp`: Generated training data files for the models (auto-generated)
- `inference_images`: Images resulting from model predictions
- `logs`: Logs of model training
- `models`: Custom model architectures
- `utils`: Auxiliary tools like dataset validation, independent of the main program
- `train.py`: Model training script
- `test.py`: Evaluates model performance on the test set
- `visualize.py`: Visualizes model performance

### Model Weights Download
[Model Weights Download](https://github.com/echonoshy/floodnet-segmentation/releases/tag/v0.1.0)

After downloading the models, verify the MD5 checksum, then place the model weights in the `exp/model_{model_name}` folder, such as `exp/model_deeplab/deeplab_best.pth`.

### Quick Start
1. Modify dataset paths in `config` and set batch size according to your hardware. Set `activated_model` to specify which model to use for training, prediction, and visualization.
2. Train the model:
   ```bash
   nohup python train.py > 2>&1 &
   ```
3. Validate model metrics on the test set:
   ```bash
   python test.py
   ```
4. Visualize results with the current model:
   ```bash
   python visualize.py
   ```

## Overall Design Approach
From the beginning, I envisioned this as not just an experiment but a fully deployable application. The plan includes considerations that may not all be implemented in this version but will be updated incrementally:

1. **Code Architecture Design:**
   - **Scalability:** A micro-framework that can be quickly adapted to different models for training, prediction, and visualization.
   - **Configurability:** Unified configuration to accommodate various models and features with minimal impact on the main program.
   - **Decoupling Functional Modules:** Separate different functions to make them more independent, increasing encapsulation and reducing redundancy.

2. **Improve Model Performance: Data, Data, Data:**
   - **Image Adjustment and Normalization:** Already implemented.
   - **Image Augmentation:** Explore advanced augmentation methods like rotation and scaling using the `albumentations` library.
   - **Incorporate Unlabeled Data:** Include unlabeled data for training with noise-student-training or rule-based selection.
   - **Save More Model Weights:** Use techniques like average_model to improve performance.

3. **Training and Deployment:**
   - **Training Acceleration with Clustering:**
   - **Model Optimization:** Quantization, distillation, and ONNX conversion.

## Experimental Process

### 1. Data Preprocessing
- **Data Cleaning:** Skipped as all data was already clean.
- **Image Adjustment:** All images and labels were resized to 256x256 pixels to match model input sizes.
- **Normalization:** Normalized images using dataset-specific means and standard deviations.
- **Image Augmentation:** Used horizontal and vertical flipping.

### 2. Evaluation Plan
- **Metrics:**
  - **Dice Coefficient:** Used as the primary evaluation metric.
  - **mIoU:** Calculate mean IoU across batches.

- **Visualization:** Qualitative comparison of model predictions against ground truth.

### 3. Model Training
Three models were trained: UNet, PSPNet, and DeepLab. PSPNet performed best on the test set. However, the Dice metric fluctuated after 10 epochs, so only the best val_dice weights and the latest weights were saved for space reasons.

![PSPNet Val Dice](https://github.com/echonoshy/floodnet-segmentation/blob/master/exp/model_pspnet/pspnet_val_dice.png)

### 4. Results Display
1. Model performance:
| Model       | Val-Dice | Test-Dice | Test-mIoU |
|-------------|----------|-----------|-----------|
| UNet        | 0.5354   | 0.4528    | 0.3234    |
| PSPNet      | 0.6387   | 0.5825    | 0.4732    |
| DeepLab V3  | 0.6258   | 0.5636    | 0.4259    |

PSPNet performed best, consistent with the FloodNet paper.

2. Visualization:
![Inference Images](https://github.com/echonoshy/floodnet-segmentation/blob/master/inference_images/model_merged_images.jpg)

## Reflection and Conclusion

### Issues
1. **Preprocessing:** Image scaling and translation could improve accuracy.
2. **Training Parameters:** No experiments with different optimizers or learning rate strategies.
3. **Class Imbalance:** Did not address class imbalance issues, affecting segmentation accuracy.
4. **Hardware Constraints:** Encountered memory issues during training.
5. **mIoU Calculation Error:** Fixed an error with mIoU calculation where missing classes caused a lower result.

### Integration with DeepForest

**DeepForest** is a crown detection tool for aerial images that focuses on object detection rather than pixel-level segmentation. However, some potential approaches could integrate it with FloodNet:

1. **Crown Detection and Segmentation:**
   - Use DeepForest to identify tree regions and improve segmentation of vegetation.

2. **Object Detection and Data Annotation:**
   - Use DeepForest to pre-label FloodNet images.

3. **Feature Fusion:**
   - Use detection features as additional inputs for semantic segmentation.

4. **Fine-tuning and Expansion:**
   - Fine-tune DeepForest models for FloodNet data.

### Conclusion

The code explored different semantic segmentation models on FloodNet, providing insights into model behavior and challenges in flood scenes. Future work should focus on improving preprocessing techniques and exploring model integration to enhance segmentation accuracy.