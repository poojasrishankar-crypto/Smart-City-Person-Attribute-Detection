# Smart City Person Attribute Detection

## Project Overview
This project performs multi-label person attribute recognition for smart city surveillance.

The system predicts multiple attributes from a single image such as:
- Upper body color
- Lower body color
- Clothing type
- Sleeve length
- Carry objects
- Accessories
- Footwear
- Pose
- View (front/back)

This project was developed as part of a challenge conducted by Vehant Research Lab.

---

## Dataset
- ~701 images
- 50 binary attributes
- Format:
image_name attr1 attr2 attr3 ... attr50

Dataset is not uploaded due to license restrictions.

---

## Tools Used
- Google Colab (GPU training)
- PyTorch
- OpenCV
- Roboflow (Annotation)

---

## Model
- ResNet50 (Transfer Learning)
- Output layer modified to 50 attributes
- Loss: BCEWithLogitsLoss
- Optimizer: Adam

---

## Author
Pooja Shankar
