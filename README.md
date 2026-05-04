# 🚀 Deep Learning — Neural Networks & Computer Vision

<div align="center">

[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Stars](https://img.shields.io/github/stars/Vaishnavi-Dubey/Deep-Learning.svg?style=for-the-badge)](https://github.com/Vaishnavi-Dubey/Deep-Learning/stargazers)

![Python](https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</div>

> Implementation of advanced neural network architectures — **CNNs**, **RNNs**, **VGG16**, **Face Recognition**, **Object Detection**, **Hate Speech Detection**, and **Medical Diagnosis** — using PyTorch and TensorFlow.

---

## ✨ Key Features

- 🧠 **Deep Learning Fundamentals** — Core neural network concepts and implementations
- 📸 **Convolutional Neural Networks** — Image classification and feature extraction
- 👤 **Face Recognition** — CNN-based facial recognition model
- 🔍 **Object Detection** — Real-time object detection implementations
- 🏥 **Medical Diagnoser** — AI-assisted medical diagnosis from clinical data
- 🛡️ **Hate Speech Detection** — NLP-based toxic content classification
- 🔄 **Recurrent Neural Networks** — Sequence modeling for temporal data
- 🏗️ **VGG16 Architecture** — Transfer learning with pre-trained VGG16 models

---

## 🧠 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Deep Learning** | PyTorch, TensorFlow, Keras |
| **Language** | Python 3.x |
| **Architecture** | CNN, RNN, VGG16, Transfer Learning |
| **Computer Vision** | OpenCV, PIL |
| **NLP** | NLTK, Word Embeddings |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Notebooks** | Jupyter Notebook |

---

## 🏗️ Architecture / How It Works

Each notebook implements a complete deep learning pipeline:

```
Data Loading → Preprocessing → Model Architecture → Training Loop
→ Validation → Evaluation → Inference
```

### Models Implemented

| Model | Application | Architecture |
|-------|------------|--------------|
| **CNN** | Image Classification | Conv2D → Pool → FC |
| **VGG16** | Transfer Learning | Pre-trained VGG16 + Custom Head |
| **Face Recognition** | Facial ID | CNN with Siamese/Triplet approach |
| **Object Detection** | Multi-class Detection | Detection Pipeline |
| **RNN** | Sequence Processing | LSTM/GRU Cells |
| **Medical Diagnoser** | Clinical Prediction | Custom DNN |
| **Hate Speech** | Text Classification | Embedding → RNN/CNN |

---

## 📂 Project Structure

```
Deep-Learning/
├── DeepLearning.ipynb                  # Neural network fundamentals
├── Convulation Neural Network.ipynb    # CNN from scratch
├── VGG16.ipynb                         # VGG16 architecture study
├── VGG16 CNN.ipynb                     # VGG16 transfer learning
├── Face_Recognition_CNN_model.ipynb    # Face recognition system
├── ObjectDetection.ipynb               # Object detection pipeline
├── RecurrentNeuralNetwork.ipynb        # RNN/LSTM implementations
├── Medical_Diagnoser.ipynb             # Medical diagnosis AI
├── Hate_Speech_Detection.ipynb         # Toxic content classifier
└── README.md
```

---

## ⚙️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/Vaishnavi-Dubey/Deep-Learning.git
cd Deep-Learning

# Install dependencies
pip install torch torchvision tensorflow keras
pip install pandas numpy matplotlib seaborn opencv-python jupyter

# Launch Jupyter Notebook
jupyter notebook
```

---

## ▶️ Usage

1. Open any `.ipynb` notebook in Jupyter Notebook or Google Colab
2. Run cells sequentially — each notebook is self-contained
3. Modify hyperparameters to experiment with different configurations
4. GPU recommended for VGG16 and larger models

---

## 📈 Impact / Learning / Highlights

- 🧠 **9 Distinct Architectures** — Covers the breadth of modern deep learning
- 🔬 **Theory to Practice** — Each notebook bridges mathematical concepts with working code
- 🏥 **Real-World Applications** — Medical diagnosis, face recognition, and content moderation
- 📊 **Dual Framework** — Implementations in both PyTorch and TensorFlow
- 🎯 **Production Patterns** — Transfer learning, data augmentation, and model evaluation best practices

---

## 🤝 Contributing

Contributions are welcome! Add new architectures, optimize existing models, or improve documentation.

---

## 📜 License

This project is licensed under the **MIT License**.

---

<p align="center">
  <b>Built with ❤️ by <a href="https://github.com/Vaishnavi-Dubey">Vaishnavi Dubey</a></b>
</p>
