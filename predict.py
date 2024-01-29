# predict.py
import argparse
from PIL import Image
import json
import torch
from torch import optim
from torchvision import models 
from train_utils import predict

def main():
    parser = argparse.ArgumentParser(description="Predict flower name from an image.")
    parser.add_argument("input", help="Path to the input image")
    parser.add_argument("checkpoint", help="Path to the model checkpoint")
    parser.add_argument("--top_k", dest="top_k", type=int, default=5, help="Return top K most likely classes")
    parser.add_argument("--category_names", dest="category_names", default="cat_to_name.json", help="Mapping of categories to real names")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")

    args = parser.parse_args()

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    checkpoint = torch.load(args.checkpoint)
    model_architecture = checkpoint['model_architecture']
    if model_architecture.lower() == 'mobilenetv3':
        model = models.mobilenet_v3_small(pretrained=True)
    elif model_architecture.lower() == 'vgg':
        model = models.vgg16(pretrained=True)
    elif model_architecture.lower() == 'resnet':
                model = models.resnet50(pretrained=True)
    else:
        raise ValueError(f"Unsupported model architecture: {model_architecture}")
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    top_probabilities, top_classes = predict(args.input, model, device,args.top_k)

    class_names = [cat_to_name[class_label] for class_label in top_classes]
    class_label = cat_to_name[args.input.split('\\')[-2]]
    print(args.input.split('\\'))
    print("Actual Class: ",class_label)
    for i in range(args.top_k):
        print(f"Top {i+1} Prediction: {class_names[i]} with Probability: {top_probabilities[i]:.3f}")

if __name__ == "__main__":
    main()
