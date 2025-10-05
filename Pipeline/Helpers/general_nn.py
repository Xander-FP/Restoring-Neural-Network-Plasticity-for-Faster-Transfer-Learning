import torch

def general_validate(self, data_loader, loss_function):
    correct = 0
    loss = 0
    total = 0
    self.model.eval()
    with torch.no_grad():
        for images, labels, _ in data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss += loss_function(outputs, labels).item()

            # Calculating accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            del images, labels, outputs
    return loss/len(data_loader), 100 * correct / total

def general_batch_lightning(lit_model, batch, criterion):
    correct = 0
    total = 0
    # Forward pass
    images, labels, _ = batch
    outputs = lit_model.encoder(images)
    loss = criterion(outputs, labels)

    # Calculating accuracy
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

    return loss, 100*correct/total, len(labels)

def general_batch_lightning_autoencode(lit_model, batch, criterion):
    # Forward pass
    images, _, _ = batch
    outputs = lit_model.encoder(images)
    loss = criterion(outputs, images)

    return loss, 100*1/loss.item(), len(images)
        
def general_train(self, train_loader, loss_function, optimizer):
    train_loss = 0
    correct = 0
    total = 0
    for images, labels, _ in train_loader:  
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass
        outputs = self.model(images)
        loss = loss_function(outputs, labels)

        # Calculating accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update loss
        train_loss += loss.item()
    return train_loss/len(train_loader), 100 * correct / total