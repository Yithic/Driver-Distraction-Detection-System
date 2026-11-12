# Comparative Analysis of Deep Learning Architectures for Driver Distraction Detection

An end-to-end computer vision pipeline comparing four deep learning architectures—a Custom 2D CNN, MobileNetV2, ResNet50, and a Vision Transformer (`vit_b16_fe`)—on the **State Farm Distracted Driver Detection** dataset, culminating in a real-time macOS OpenCV webcam inference deployment.

## 📌 Overview
Driver distraction is a leading cause of automotive accidents. This project automates driver posture and behavior classification across 10 distinct categories from dashboard camera feeds. Models were trained on Google Colab (NVIDIA T4 GPU) using a two-phase fine-tuning strategy with `EarlyStopping`, `ReduceLROnPlateau`, and dataset-safe augmentation (strictly omitting horizontal flips to preserve left-hand vs. right-hand ground-truth labels).

## 🎯 Dataset Classes (`c0` – `c9`)
Trained on 2D RGB dashboard images from the State Farm Distracted Driver dataset:
* **`c0`**: Safe driving
* **`c1`**: Texting – right hand
* **`c2`**: Talking on the phone – right hand
* **`c3`**: Texting – left hand
* **`c4`**: Talking on the phone – left hand
* **`c5`**: Operating the radio
* **`c6`**: Drinking
* **`c7`**: Reaching behind
* **`c8`**: Hair and makeup
* **`c9`**: Talking to passenger

## 📊 Experimental Results

| Model | Top-1 Accuracy | Log Loss | F1-Score | Multiclass OvR AUC | Total Parameters | Model Size | Latency / Frame | Throughput | FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Custom 2D CNN** | 99.31% | 0.0304 | 0.9929 | 0.9999 | 11.17 M | 127.89 MB | 92.29 ms | 10.84 FPS | 0.6176 |
| **MobileNetV2** | 99.15% | 0.0339 | 0.9914 | 0.9997 | 2.27 M | 23.57 MB | 87.43 ms | 11.44 FPS | 0.6462 |
| **ResNet50** | **99.53%** | **0.0222** | **0.9952** | 0.9998 | 23.61 M | 205.16 MB | 103.81 ms | 9.63 FPS | 0.6484 |
| **ViT (`vit_b16_fe`)** | 23.60% | 2.0880 | 0.1775 | 0.7522 | 85.81 M | 327.48 MB | 109.71 ms | 9.11 FPS | 1.0000 |

### Key Takeaways
* **Highest Accuracy:** **ResNet50** achieved the highest validation accuracy (`99.53%`) and lowest log loss (`0.0222`).
* **Best for Edge/Real-Time Inference:** **MobileNetV2** delivered `99.15%` accuracy with the smallest footprint (`23.57 MB`) and fastest local throughput (`11.44 FPS` / `87.43 ms` latency).

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** TensorFlow, Keras, TensorFlow Hub
* **Training Environment:** Google Colab (NVIDIA T4 GPU)
* **Deployment & Vision:** OpenCV (`cv2`), macOS

## 📂 Repository Structure
```text
├── Distraction_Detection_Detection_Comparati...ipynb  # Data pipeline, 2-phase fine-tuning & evaluation
├── common.py                                          # Shared OpenCV webcam capture, preprocessing & overlay utilities
├── run_custom_cnn.py                                  # Live webcam inference using Custom 2D CNN
├── run_mobilenet.py                                   # Live webcam inference using MobileNetV2
├── run_resnet.py                                      # Live webcam inference using ResNet50
├── run_vit.py                                         # Live webcam inference using Vision Transformer (ViT)
└── README.md
