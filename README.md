<h1>📡 Multi-Class Airborne Detection System (YOLOv11)</h1>

This project is a real-time airborne object detection system designed to identify and classify drones, birds, airplanes, and helicopters with high accuracy.  
Built using YOLOv11 and optimized for NVIDIA Jetson devices, the system delivers fast and reliable detection in outdoor, dynamic environments.

---

<h2>🌟 What This System Does</h2>

### ✔ Detects Multiple Airborne Objects in Real Time
- Drones (various sizes and angles)
- Birds (different species & wing patterns)
- Airplanes
- Helicopters

### ✔ Uses Enhanced Deep-Learning Techniques
- YOLOv11 anchor-free architecture  
- Quantum-inspired feature extraction  
- Neuromorphic motion-aware enhancement  
- Wavelet-based multi-resolution feature boosting  

### ✔ Designed for Edge AI Devices
- Runs smoothly on NVIDIA Jetson Nano / Xavier / Orin  
- GPU-optimized using TensorRT  
- Low latency + high FPS performance  

### ✔ Produces Detailed Analytical Metrics
- Precision, Recall, F1-Score  
- mAP (mean average precision)  
- Confusion Matrix  
- Confidence curves  

---

<h2>🧠 Why This Project Exists</h2>

Airborne threats—especially drones—are becoming more common near:

- Airports  
- Military zones  
- Critical infrastructure  
- Wildlife monitoring sites  

Traditional radar/RF systems are:

- Expensive  
- Hard to deploy  
- Poor at detecting small or low-flying drones  

This deep-learning system enables:

- Early detection  
- High-accuracy classification  
- Real-time tracking  
- Low-cost, compact edge deployment  

---

<h2>🗂 Dataset Used</h2>

### Contains Images Of:
- Drone images  
- Bird flight images  
- Airplanes at multiple altitudes  
- Helicopters in motion  

### **Preprocessing Includes**
- Resizing  
- Normalization  
- Rotation & flipping  
- Scaling  
- Brightness adjustments  

These steps improve generalization under varied outdoor conditions.

---

<h2>🔌 Core Technology Used</h2>

### 🧠 YOLOv11 Object Detector
- Anchor-free detection  
- Faster & more accurate than previous YOLO versions  
- Excellent for small airborne objects  

### ⚡ Hybrid Feature Enhancements
- Quantum-inspired convolution  
- Neuromorphic event-aware processing  
- Wavelet multi-resolution mapping  

### 🖥 Edge Deployment on Jetson
- TensorRT acceleration  
- FP16 precision optimization  
- Real-time video feed processing  

---

<h2>🛠 How It Works</h2>

### **1️⃣ Video Stream Input**
The system processes video frames from:
- Webcam  
- Drone camera  
- RTSP stream  
- Stored video file  

### **2️⃣ Real-Time Detection Pipeline**
Each frame undergoes:
- Preprocessing  
- Feature extraction  
- Multi-scale detection heads  
- Classification + bounding box assignment  

### **3️⃣ Enhanced Recognition Accuracy**
Hybrid enhancement modules help the system:
- Distinguish drones from birds  
- Detect small distant objects  
- Handle motion blur  
- Perform reliably in cluttered scenes  

### **4️⃣ Output Visualization**
The system displays:
- Bounding boxes  
- Class labels  
- Confidence values  
