# Smart City Person Attribute Detection

Multi-label person attribute recognition system for smart city surveillance using ResNet50 and PyTorch.

---

## 📌 Project Overview

This project detects multiple attributes of a person from an image such as:

- Upper & Lower body clothing color
- Clothing type (Jacket, Kurta, Saree, Shirt, etc.)
- Sleeves length
- Carry items (Handbag, Backpack)
- Accessories (Headgear)
- Footwear (Shoes, Sandals, Slippers)
- Pose (Standing, Sitting, Lying)
- View orientation (Front / Back)

The system is built using **Transfer Learning with ResNet50** and trained using **BCEWithLogitsLoss** for multi-label classification.

---

## 🧠 Model Architecture

- Backbone: ResNet50 (Pretrained on ImageNet)
- Loss Function: BCEWithLogitsLoss
- Activation Function: Sigmoid
- Optimizer: Adam
- Framework: PyTorch
- Task Type: Multi-Label Classification

---

## 📊 Model Performance

| Metric        | Value (Approx) |
|--------------|----------------|
| Training Loss | ~0.35 |
| Validation Loss | ~0.32 |
| Accuracy | ~0.84 |
| F1 Score | ~0.81 |
| Epochs | 5 |

---

## 🔍 Sample Inference Output

Example prediction on test image:

Detected Attributes:
- Foot_slippers (0.69)
- POSE_standing (0.95)
- VIEW_front (0.52)

Confidence values represent probability scores from the sigmoid output layer.

---

## 🏗 Project Structure

```
Smart-City-Person-Attribute-Detection/
│── dataset.py
│── model.py
│── train.py
│── inference.py
│── model.pth (generated after training)
│── train.txt
│── README.md
│── LICENSE
│── .gitignore
```

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies

```
pip install torch torchvision opencv-python scikit-learn
```

### 2️⃣ Train the Model

```
python train.py
```

This will:
- Train the model
- Calculate validation metrics
- Save trained weights as `model.pth`

### 3️⃣ Run Inference

```
python inference.py
```

This will:
- Load trained model
- Predict attributes
- Print confidence scores

---

## 📦 Dataset

Dataset is not uploaded due to size limitations.

Before training, ensure:

- `train.txt` is present
- `images/` folder contains all dataset images

Place them inside the project directory.

---

## 🎯 Key Concepts Used

- Transfer Learning
- Multi-Label Classification
- Sigmoid Activation
- BCEWithLogitsLoss
- F1 Score Evaluation
- Threshold-based Attribute Detection

---

## 👩‍💻 Author

Pooja Shankar  
Third Year Electronics and Communication Engineering  
