🛩️ Multi-Class Airborne Entity Detection using YOLOv11

This project presents a real-time airborne object detection system capable of identifying drones, birds, airplanes, and helicopters using the YOLOv11 deep-learning framework. The model is optimized for NVIDIA Jetson edge devices, enabling fast, low-latency inference on live video streams captured using a standard webcam. The system is designed for applications in airspace monitoring, aerial surveillance, UAV awareness, and safety analysis.

🌟 Key Features

Multi-class airborne detection (Drone, Bird, Airplane, Helicopter)

YOLOv11 anchor-free architecture for improved accuracy

Hybrid feature enhancement (quantum-inspired, neuromorphic, wavelet-based methods)

Edge deployment on NVIDIA Jetson for offline, real-time performance

Evaluation metrics: Precision, Recall, F1-score, mAP, Confusion Matrix

Robust performance in diverse environmental conditions

🧠 Why This System Matters

Traditional radar and RF-based surveillance systems struggle with:

Small or low-flying objects

High infrastructure cost

Inefficiency in complex environments

This work provides a low-cost, scalable, AI-driven alternative using camera-based detection.
YOLOv11 delivers high performance with minimal computational overhead, making it ideal for embedded platforms and real-time surveillance.

🛠️ Methodology Overview

Dataset Creation
Combined aerial datasets covering drones, birds, airplanes, and helicopters, along with webcam-captured samples.

Preprocessing & Augmentation
Resizing, normalization, noise reduction, rotation, flipping, scaling, and brightness adjustments.

YOLOv11 Detection Framework
Anchor-free detection improves performance on small, fast-moving airborne objects.

Hybrid Feature Enhancement

Quantum-inspired convolution

Neuromorphic-inspired event modeling

Wavelet-based multi-resolution feature extraction

Edge Deployment
Optimized and deployed on NVIDIA Jetson with GPU acceleration for real-time inference.

🧩 System Architecture

The architecture extends YOLOv11 using:

Multi-stream feature extraction

Wavelet downsampling (HWD)

Spatial pyramid pooling (SPPF)

Attention-based feature selection

Multi-scale detection heads

This improves robustness in cluttered backgrounds and supports variable object sizes.
