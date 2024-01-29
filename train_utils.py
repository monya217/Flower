# train_utils.py
import torch
import torch
from PIL import Image
import numpy as np

def train_model(model, trainloader, valloader, criterion, optimizer, device, epochs=10, print_every=10):
    model.to(device)

    for epoch in range(epochs):
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

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(valloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(valloader):.3f}")
                running_loss = 0
                model.train()

def save_checkpoint(model, optimizer, train_data, file_path='checkpoint.pth'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': train_data.class_to_idx,
        'classifier': model.classifier
    }

    torch.save(checkpoint, file_path)



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image)

    # Resize the image
    size = 256
    shortest_side = min(image.size)
    aspect_ratio = image.width / image.height if image.width > image.height else image.height / image.width
    new_size = [int(size * aspect_ratio), size] if image.width > image.height else [size, int(size * aspect_ratio)]
    image = image.resize(new_size)

    # Crop the center of the image
    crop_size = 224
    left = (image.width - crop_size) / 2
    top = (image.height - crop_size) / 2
    right = (image.width + crop_size) / 2
    bottom = (image.height + crop_size) / 2
    image = image.crop((left, top, right, bottom))

    # Convert to numpy array
    np_image = np.array(image) / 255.0

    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Transpose dimensions
    np_image = np_image.transpose((2, 0, 1))

    # Convert to PyTorch tensor
    torch_image = torch.from_numpy(np_image).float()

    return torch_image
def predict(image, model,device, top_k = 5):
    image = process_image(image)

    image = image.to(device)

    # Move the model to the same device as the image
    model = model.to(device)
    # Ensure the model is in evaluation mode
    model.eval()

    # Perform a forward pass to get the logits
    with torch.no_grad():
        logits = model(image.unsqueeze(0))

    # Calculate the probabilities using softmax
    probabilities = torch.exp(logits)

    # Get the top K probabilities and classes
    top_probabilities, top_classes = probabilities.topk(top_k, dim=1)

    # Convert indices to class labels using class_to_idx
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_classes[0].tolist()]

    return top_probabilities[0].tolist(), top_classes


