import torch
from torchvision import transforms
from PIL import Image

# Same transforms used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_image(model, image_path, device="cpu"):
    """
    Predict whether the skin lesion is Benign or Malignant
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    label = "Malignant" if predicted.item() == 1 else "Benign"
    confidence_score = confidence.item() * 100

    return label, confidence_score, probabilities[0].cpu().numpy()