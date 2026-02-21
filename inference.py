import torch
import cv2
import numpy as np
import os
from model import get_model

# ---------------- PATHS ----------------
MODEL_PATH = "model.pth"
IMG_PATH = r"C:\Users\USER\Documents\Smart-City-Person-Attribute-Detection\images\530.jpg"
THRESHOLD = 0.5   # You can change this (0.4, 0.6 etc.)
# ---------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Attribute Names (49) --------
attribute_names = [

    # Upper Body Color
    "UBCOLOR_black", "UBCOLOR_blue", "UBCOLOR_green",
    "UBCOLOR_orange", "UBCOLOR_red", "UBCOLOR_white",
    "UBCOLOR_yellow", "UBCOLOR_mix", "UBCOLOR_other",

    # Lower Body Color
    "LBCOLOR_black", "LBCOLOR_blue", "LBCOLOR_green",
    "LBCOLOR_red", "LBCOLOR_yellow", "LBCOLOR_white",
    "LBCOLOR_orange", "LBCOLOR_other", "LBCOLOR_mix",

    # Upper Body Clothing
    "UB_jacket", "UB_kurta", "UB_other",
    "UB_Saree", "UB_Shirt", "UB_Suitwomen",
    "UB_tshirt", "UB_sweater",

    # Lower Body Clothing
    "LB_Leggings", "LB_Salwar", "LB_Shorts",
    "LB_Trousers", "LB_Jeans", "LB_Saree",
    "LB_other",

    # Sleeves Length
    "SLEEVES_long", "SLEEVES_short", "SLEEVES_none",

    # Carry
    "Carry_handbag", "Carry_backpack", "Carry_other",

    # Accessory
    "Acc_headgear",

    # Footwear
    "Foot_Sandals", "Foot_shoes", "Foot_slippers",

    # Pose
    "POSE_sitting", "POSE_lying", "POSE_standing",

    # View
    "VIEW_back", "VIEW_front"
]
# ---------------------------------------


# Load model
model = get_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Check image exists
if not os.path.exists(IMG_PATH):
    raise FileNotFoundError(f"Image not found at: {IMG_PATH}")

# Read image
image = cv2.imread(IMG_PATH)

if image is None:
    raise ValueError("OpenCV failed to read image.")

# Preprocess
image = cv2.resize(image, (224, 224))
image = image / 255.0
image = torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0)
image = image.to(device)

# -------- Inference --------
with torch.no_grad():
    outputs = model(image)
    probabilities = torch.sigmoid(outputs)

print("\n========== Prediction Result ==========")
print(f"Threshold: {THRESHOLD}")
print("---------------------------------------")

found = False

for i in range(len(attribute_names)):
    confidence = probabilities[0][i].item()
    if confidence > THRESHOLD:
        print(f"{attribute_names[i]:20} | Confidence: {confidence:.2f}")
        found = True

if not found:
    print("No attributes detected above threshold.")

print("=======================================\n")