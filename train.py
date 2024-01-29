# train.py
import argparse
from torchvision import models
from torch import nn, optim
from torchvision import datasets, transforms
import torch

def main():
    parser = argparse.ArgumentParser(description="Train a new network on a dataset and save the model as a checkpoint.")
    parser.add_argument("data_directory", help="Path to the data directory")
    parser.add_argument("--save_dir", dest="save_directory", default="checkpoint.pth", help="Directory to save checkpoints")
    parser.add_argument("--arch", dest="arch", default="vgg16", help="Choose architecture (vgg16, densenet121, etc.)")
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=0.001, help="Set learning rate")
    parser.add_argument("--hidden_units", dest="hidden_units", type=int, default=512, help="Set number of hidden units")
    parser.add_argument("--epochs", dest="epochs", type=int, default=1, help="Set number of epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")

    args = parser.parse_args()

    # Define data transforms and load data
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    train_dir = args.data_directory + '/train'
    val_dir = args.data_directory + '/valid'
    
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    val_data = datasets.ImageFolder(val_dir, transform=test_transforms)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=True)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    # Load pre-trained model
    model = getattr(models, args.arch)(pretrained=True)

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define custom classifier
    input_size = model.classifier[0].in_features
    classifier = nn.Sequential(
        nn.Linear(input_size, args.hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(args.hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier

    # Train the model
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    print_every = 10
    for epoch in range(args.epochs):
        steps = 0
        running_loss = 0

        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for inputs, labels in valloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{args.epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(valloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(valloader):.3f}")
                running_loss = 0
                model.train()
    # Save the checkpoint
    checkpoint = {
        'model_architecture': type(model).__name__, 
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': train_data.class_to_idx,
        'classifier': model.classifier
    }

    torch.save(checkpoint, args.save_directory)
if __name__ == "__main__":
    main()
