# Comparative Analysis of Deep Learning Architectures for Driver Distraction Detection

An end-to-end computer vision pipeline comparing four deep learning architectures—a Custom 2D CNN, MobileNetV2, ResNet50, and a Vision Transformer (`vit_b16_fe`)—on the **State Farm Distracted Driver Detection** dataset, culminating in a real-time macOS OpenCV webcam inference deployment[cite: 4, 9].

## 📌 Overview
Driver distraction is a leading cause of automotive accidents[cite: 6]. This project automates driver posture and behavior classification across 10 distinct categories from dashboard camera feeds[cite: 4, 7]. Models were trained on Google Colab (NVIDIA T4 GPU) using a two-phase fine-tuning strategy with `EarlyStopping`, `ReduceLROnPlateau`, and dataset-safe augmentation (strictly omitting horizontal flips to preserve left-hand vs. right-hand ground-truth labels)[cite: 6, 8, 9].

## 🎯 Dataset Classes (`c0` – `c9`)
Trained on 2D RGB dashboard images from the State Farm Distracted Driver dataset[cite: 7]:
* **`c0`**: Safe driving[cite: 7]
* **`c1`**: Texting – right hand[cite: 7]
* **`c2`**: Talking on the phone – right hand[cite: 7]
* **`c3`**: Texting – left hand[cite: 7]
* **`c4`**: Talking on the phone – left hand[cite: 7]
* **`c5`**: Operating the radio[cite: 7]
* **`c6`**: Drinking[cite: 7]
* **`c7`**: Reaching behind[cite: 7]
* **`c8`**: Hair and makeup[cite: 7]
* **`c9`**: Talking to passenger[cite: 7]

## 📊 Experimental Results

| Model | Top-1 Accuracy | Log Loss | F1-Score | Multiclass OvR AUC | Total Parameters | Model Size | Latency / Frame | Throughput | FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Custom 2D CNN** | 99.31%[cite: 9] | 0.0304[cite: 9] | 0.9929[cite: 9] | 0.9999[cite: 9] | 11.17 M[cite: 9] | 127.89 MB[cite: 9] | 92.29 ms[cite: 9] | 10.84 FPS[cite: 9] | 0.6176[cite: 9] |
| **MobileNetV2** | 99.15%[cite: 9] | 0.0339[cite: 9] | 0.9914[cite: 9] | 0.9997[cite: 9] | 2.27 M[cite: 9] | 23.57 MB[cite: 9] | 87.43 ms[cite: 9] | 11.44 FPS[cite: 9] | 0.6462[cite: 9] |
| **ResNet50** | **99.53%**[cite: 9] | **0.0222**[cite: 9] | **0.9952**[cite: 9] | 0.9998[cite: 9] | 23.61 M[cite: 9] | 205.16 MB[cite: 9] | 103.81 ms[cite: 9] | 9.63 FPS[cite: 9] | 0.6484[cite: 9] |
| **ViT (`vit_b16_fe`)** | 23.60%[cite: 9] | 2.0880[cite: 9] | 0.1775[cite: 9] | 0.7522[cite: 9] | 85.81 M[cite: 9] | 327.48 MB[cite: 9] | 109.71 ms[cite: 9] | 9.11 FPS[cite: 9] | 1.0000[cite: 9] |

### Key Takeaways
* **Highest Accuracy:** **ResNet50** achieved the highest validation accuracy (`99.53%`) and lowest log loss (`0.0222`)[cite: 9].
* **Best for Edge/Real-Time Inference:** **MobileNetV2** delivered `99.15%` accuracy with the smallest footprint (`23.57 MB`) and fastest local throughput (`11.44 FPS` / `87.43 ms` latency)[cite: 9].

## 🛠️ Tech Stack
* **Language:** Python[cite: 6]
* **Deep Learning:** TensorFlow, Keras, TensorFlow Hub[cite: 6]
* **Training Environment:** Google Colab (NVIDIA T4 GPU)[cite: 6]
* **Deployment & Vision:** OpenCV (`cv2`), macOS[cite: 6]

## 📂 Repository Structure
```text
├── Distraction_Detection_Detection_Comparati...ipynb  # Data pipeline, 2-phase fine-tuning & evaluation[cite: 13]
├── common.py                                          # Shared OpenCV webcam capture, preprocessing & overlay utilities[cite: 6, 9, 13]
├── run_custom_cnn.py                                  # Live webcam inference using Custom 2D CNN[cite: 4, 13]
├── run_mobilenet.py                                   # Live webcam inference using MobileNetV2[cite: 4, 13]
├── run_resnet.py                                      # Live webcam inference using ResNet50[cite: 4, 13]
├── run_vit.py                                         # Live webcam inference using Vision Transformer (ViT)[cite: 4, 13]
└── README.md
