import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("data", transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

print("Classes:", dataset.classes)

model = models.efficientnet_b0(weights="DEFAULT")
model.classifier[1] = nn.Linear(1280, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for images, labels in loader:
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training step complete!")

def predict(image_path):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)

    return probs.cpu()

test_image = dataset.samples[0][0]

probs = predict(test_image)

classes = dataset.classes
pred = probs.argmax(dim=1).item()
confidence = probs[0][pred].item()

print(f"Prediction: {classes[pred]} ({confidence*100:.2f}%)")