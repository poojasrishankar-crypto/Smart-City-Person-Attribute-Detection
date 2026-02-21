import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset import PersonAttributeDataset
from model import get_model
import os

# --------- CHANGE THIS PATH ----------
DATA_PATH = r"C:\Users\USER\Documents\Smart-City-Person-Attribute-Detection"
IMG_DIR = os.path.join(DATA_PATH, "images")
TXT_FILE = os.path.join(DATA_PATH, "train.txt")
# -------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = PersonAttributeDataset(TXT_FILE, IMG_DIR)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = get_model().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "model.pth")
print("Training complete. Model saved.")