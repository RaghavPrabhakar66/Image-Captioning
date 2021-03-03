import torch
from tqdm import tqdm

# training function
def train(model, dataloader, optimizer, criterion, train_data, device):
    # Switch model to training mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
    model.train()

    train_running_loss = 0.0
    train_running_correct = 0
    
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        # Load the input features and labels from the training dataset
        image, target_caption = data['image'].to(device), data['caption'].to(device)

        # Reset the gradients to 0 for all learnable weight parameters
        optimizer.zero_grad()

        # Forward pass: Pass image data from training dataset, make predictions about class image belongs to (0-4 in this case)
        predicted_caption = model(data)

        # Define our loss function, and compute the loss
        loss = criterion(predicted_caption, target_caption)

        train_running_loss += loss.item()
        _, preds = torch.max(predicted_caption.data, 1)
        train_running_correct += (preds == target_caption).sum().item()

        # Backward pass: compute the gradients of the loss w.r.t. the model's parameters
        loss.backward()

        # Update the neural network weights
        optimizer.step()
        
    train_loss = train_running_loss/len(dataloader.dataset)
    train_accuracy = 100. * train_running_correct/len(dataloader.dataset)

    return train_loss, train_accuracy