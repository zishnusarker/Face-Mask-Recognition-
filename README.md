<h1 align="center"> Face Mask Recognition</h1>

<p align="center">
  <strong>A real-time face mask detection system using PyTorch and ResNet transfer learning - with webcam-based live detection and a Pygame audio alarm that triggers when no mask is detected.</strong>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="#"><img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"></a>
  <a href="#"><img src="https://img.shields.io/badge/Transfer%20Learning-ResNet-blue?style=for-the-badge" alt="Transfer Learning"></a>
  <a href="#"><img src="https://img.shields.io/badge/Category-Computer%20Vision-red?style=for-the-badge" alt="Category"></a>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-features">Features</a> •
  <a href="#-how-it-works">How It Works</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-architecture">Architecture</a>
</p>

---

## 📖 Overview

A **real-time face mask detection system** that uses a webcam to determine whether a person is wearing a face mask or not. Built with **PyTorch** using **ResNet transfer learning** (ResNet50 and ResNet101 variants), the system classifies each frame into two classes - `with_mask` or `without_mask` - and triggers an **audio alarm** via Pygame when no mask is detected.

This project was developed as an **ADA (Algorithm Design and Analysis) course project** and demonstrates practical application of deep learning and computer vision for real-world safety monitoring during the COVID-19 era.

> **Real-world use case:** Automated mask compliance monitoring in offices, hospitals, airports, and public spaces.

---

## ✨ Features

- 🎥 **Real-time webcam detection** - Continuous frame-by-frame classification
- 🔊 **Audio alarm** - Pygame plays `alarm.wav` when no mask is detected
- 🧠 **Two ResNet variants** - ResNet50 (inference.py) and ResNet101 (label_detect.py)
- 📸 **Single image inference** - Classify individual images without webcam
- 🏋️ **Training notebook** - Jupyter notebook for model training from scratch
- 🗂️ **Data generator** - Scripts to prepare training dataset
- 📊 **ImageNet preprocessing** - Standard normalization for transfer learning
- 🖥️ **Live text overlay** - Displays classification label on video feed

---

## ⚙️ How It Works

### Detection Pipeline

```
Webcam Feed (OpenCV)
        │
   Frame captured
        │
        ▼
┌──────────────────────────┐
│    Image Preprocessing   │
│                          │
│  • BGR → RGB conversion  │
│  • Resize to 256px       │
│  • Center crop to 224px  │
│  • Normalize (ImageNet)  │
│    [0.485, 0.456, 0.406] │
│    [0.229, 0.224, 0.225] │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│   ResNet50 / ResNet101   │
│   (Transfer Learning)    │
│                          │
│   Pre-trained on ImageNet│
│   Fine-tuned on mask data│
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│      Classification      │
│                          │
│   0 → "with_mask"  ✅    │
│   1 → "without_mask" ❌  │
└────────┬─────────────────┘
         │
    ┌────┴────────────┐
    │                  │
    ▼                  ▼
 with_mask         without_mask
 → "No Beep"       → alarm.wav 🔊
 → Label on frame  → Label on frame
```

---

## 🛠 Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Deep Learning** | PyTorch | Model training and inference |
| **Pre-trained Model** | ResNet50 / ResNet101 | Transfer learning backbone |
| **Computer Vision** | OpenCV (cv2) | Webcam capture and frame processing |
| **Image Processing** | torchvision transforms | Resize, crop, normalize |
| **Audio** | Pygame mixer | Alarm sound playback |
| **Data Science** | NumPy, Pandas, scikit-learn | Data processing and splitting |
| **Visualization** | Matplotlib | Training curves |
| **Notebook** | Jupyter | Interactive model training |
| **Image Handling** | Pillow (PIL) | Image format conversion |

### Model Details

| Property | Value |
|----------|-------|
| **Architecture** | ResNet50 / ResNet101 (ImageNet pre-trained) |
| **Input Size** | 224 × 224 × 3 (RGB) |
| **Output Classes** | 2 - `with_mask`, `without_mask` |
| **Preprocessing** | ImageNet normalization (mean/std) |
| **Saved Models** | `mask1_model_resnet50.pth`, `mask1_model_resnet101.pth` |
| **Inference Device** | CPU (GPU optional via CUDA) |

---

## 🚀 Installation

### Prerequisites
- Python 3.7+
- Webcam (for real-time detection)
- GPU recommended for training (CPU works for inference)

### Setup

```bash
# Clone the repository
git clone https://github.com/zishnusarker/Face-Mask-Recognition-.git
cd Face-Mask-Recognition-/ada/observations-master/mask_classifier

# Create virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

# Install dependencies
pip install torch torchvision opencv-python pygame numpy pandas scikit-learn matplotlib pillow jupyter
```

> **Note:** The included `requirements.txt` has pinned versions from the original development environment. If you encounter compatibility issues, use the command above to install the latest compatible versions.

---

## 📋 Usage

### 1. Real-time Webcam Detection

```bash
cd ada/observations-master/mask_classifier
python webcam_detect.py
```

This will:
- Open your webcam
- Classify each frame as `with_mask` or `without_mask`
- Display the label on the video feed
- Play `alarm.wav` when no mask is detected
- Press **Q** to quit

### 2. Single Image Inference (ResNet50)

```bash
python inference.py
```

Edit the `image_path` variable in the script to point to your image:
```python
image_path = 'path/to/your/image.jpg'
```

### 3. Single Image Inference (ResNet101)

```bash
python label_detect.py
```

Uses the ResNet101 model variant for potentially higher accuracy.

### 4. Train Your Own Model

```bash
jupyter notebook training.ipynb
```

The notebook walks through:
1. Dataset preparation
2. Data augmentation and transforms
3. ResNet transfer learning setup
4. Training loop with validation
5. Model saving as `.pth`

---

## 📁 Project Structure

```
Face-Mask-Recognition-/
├── README.md
│
└── ada/                                        # ADA course project folder
    ├── ADA Project Code.docx                   # Academic project report
    │
    └── observations-master/
        ├── README.md                           # Original readme
        │
        ├── mask_classifier/                    # Main application
        │   ├── webcam_detect.py                # Real-time webcam detection + alarm
        │   ├── label_detect.py                 # Single image inference (ResNet101)
        │   ├── inference.py                    # Single image inference (ResNet50)
        │   ├── training.ipynb                  # Model training notebook
        │   ├── alarm.wav                       # Audio alarm file
        │   ├── requirements.txt                # Python dependencies
        │   │
        │   └── Data_Generator/                 # Dataset preparation scripts
        │       ├── mask.py                     # Mask overlay generation
        │       └── loop_through_folder.py      # Batch processing
        │
        └── experiements/                       # Experimental data
            ├── data/                           # Raw training data
            └── dest_folder/                    # Processed output
```

---

## 🎓 Key Concepts Demonstrated

<details>
<summary><strong>What is Transfer Learning and why use ResNet?</strong></summary>

**Transfer learning** reuses a model trained on a large dataset (ImageNet - 1.2M images, 1000 classes) for a new task. ResNet's pre-trained layers already know how to detect visual features (edges, textures, shapes, faces). We only fine-tune the final classification layer for our 2-class task (mask/no mask), which is much faster and more accurate than training from scratch.

</details>

<details>
<summary><strong>Why ResNet specifically?</strong></summary>

**ResNet** (Residual Networks) introduced skip connections that solve the vanishing gradient problem in deep networks. This allows training very deep architectures (50, 101, 152 layers) without degradation. The deeper the network, the more complex features it can learn - making ResNet ideal for fine-grained visual tasks like mask detection.

</details>

<details>
<summary><strong>Why ImageNet normalization?</strong></summary>

The preprocessing uses `mean=[0.485, 0.456, 0.406]` and `std=[0.229, 0.224, 0.225]` - these are the ImageNet dataset statistics. Since our ResNet was pre-trained on ImageNet, our input images must be normalized the same way for the learned features to work correctly. Using different normalization would produce garbage results.

</details>

<details>
<summary><strong>Why use an audio alarm?</strong></summary>

The Pygame mixer plays `alarm.wav` as an immediate auditory feedback when no mask is detected. This transforms the system from a passive classifier into an **active monitoring tool** - useful in real-world scenarios like building entrance gates, office checkpoints, or hospital corridors where visual monitoring alone isn't sufficient.

</details>

<details>
<summary><strong>ResNet50 vs ResNet101 - when to use which?</strong></summary>

**ResNet50** (50 layers) is faster for inference and has a smaller model file - better for real-time applications and edge devices. **ResNet101** (101 layers) is deeper and can learn more complex features - better for higher accuracy when computational resources aren't a constraint. This project includes both for comparison.

</details>

---

## 🔮 Future Improvements

- Add face detection (Haar Cascade or MTCNN) before classification for multi-face support
- Implement a confidence threshold - only alarm above certain probability
- Deploy as a web app with Flask/Streamlit for remote monitoring
- Add a counter for compliance statistics (% wearing masks over time)
- Convert to ONNX/TFLite for mobile or edge deployment (Raspberry Pi)
- Add support for "mask worn incorrectly" as a third class
- Implement email/SMS notifications for mask violations
- Add a dashboard for monitoring multiple camera feeds
- Train on a larger, more diverse dataset for better generalization

---

## 🌍 Real-World Applications

| Application | Description |
|-------------|-------------|
| 🏥 **Hospitals** | Enforce mask policy at entrances |
| 🏢 **Offices** | Automated compliance monitoring |
| ✈️ **Airports** | Gate and terminal safety screening |
| 🏫 **Schools** | Classroom and hallway monitoring |
| 🏪 **Retail** | Store entrance mask checking |
| 🏭 **Factories** | Safety compliance in production areas |

---

## ⚠️ Note

This project was developed during the **COVID-19 pandemic** as a practical application of computer vision for public safety. The trained models (`.pth` files) are not included in the repository due to file size - train your own using the provided notebook and dataset.

---

## 📄 License

This project is available as open source.

---

<p align="center">
  Made with ❤️ for public safety during COVID-19
</p>

<p align="center">
  <strong>Computer vision for a safer world 😷</strong>
</p>
