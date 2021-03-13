import torch
from tqdm import tqdm

# training function
def train_fit(device, model, dataloader, optimizer, criterion, dataset, ignore):
    # Switch model to training mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
    model.train()

    train_running_loss = 0.0
    train_running_correct = 0
    
    for i, data in enumerate(tqdm(dataloader, total=int(len(dataset)/dataloader.batch_size))):
        if i in ignore:
            print("Image Name : ", data['debug_img_name'], "Idx : ", data['debug_idx'])
        else:
            # Load the input features and labels from the training dataset
            target_caption = data['caption'].to(device)

            # Reset the gradients to 0 for all learnable weight parameters
            optimizer.zero_grad()

            # Forward pass: Pass image data from training dataset, make predictions about class image belongs to (0-4 in this case)
            predicted_caption = model(data)

            predicted_caption = predicted_caption.reshape(-1, predicted_caption.shape[2])
            #predicted_caption = predicted_caption.argmax(1)
            target_caption    = target_caption.reshape(-1)

            """
            batch_size = 4

            target_caption = [4, 39]
            predicted_caption = [4, 39, 8493]

            target_caption.reshape(-1) = [4 * 39]
            predicted_cation.reshape(-1, predicted_caption.shape[2]) = [4 * 39, 8493]

            """

            # Define our loss function, and compute the loss
            #print(predicted_caption.shape, target_caption.shape)
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

def validation_fit(device, model, dataloader, optimizer, criterion, dataset):
    # Switch model to training mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
    model.eval()

    val_running_loss = 0.0
    val_running_correct = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, total=int(len(dataset)/dataloader.batch_size))):
            # Load the input features and labels from the training dataset
            target_caption = data['caption'].to(device)

            # Forward pass: Pass image data from training dataset, make predictions about class image belongs to (0-4 in this case)
            predicted_caption = model(data)

            predicted_caption = predicted_caption.reshape(-1, predicted_caption.shape[2])
            #predicted_caption = predicted_caption.argmax(1)
            target_caption    = target_caption.reshape(-1)

            """
            batch_size = 4

            target_caption = [4, 39]
            predicted_caption = [4, 39, 8493]

            target_caption.reshape(-1) = [4 * 39]
            predicted_cation.reshape(-1, predicted_caption.shape[2]) = [138, 8493]

            """

            # Define our loss function, and compute the loss
            loss = criterion(predicted_caption, target_caption)

            # Define our loss function, and compute the loss
            #loss = criterion(predicted_caption.reshape(-1, predicted_caption.shape[2]), target_caption.reshape(-1))

            val_running_loss += loss.item()
            _, preds = torch.max(predicted_caption.data, 1)
            val_running_correct += (preds == target_caption).sum().item()

    val_loss = val_running_loss/len(dataloader.dataset)
    val_accuracy = 100. * val_running_correct/len(dataloader.dataset)

    return val_loss, val_accuracy