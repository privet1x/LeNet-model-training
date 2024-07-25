"""
This script trains and saves a LeNet model using the Pokémon dataset.
It also logs training history and plots accuracy and loss.
"""

import torch  # type: ignore
from torch import nn, optim  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from torchvision import datasets, transforms  # type: ignore
import pandas as pd
import matplotlib.pyplot as plt

# Disable specific pylint messages
# pylint: disable=R0914, R0903, R0913


def get_data_loaders(train_dir, test_dir, img_height=32, img_width=32, batch_size=32):
    """Create DataLoader for train, validation, and test datasets.

    Args:
        train_dir (str): Path to the training data directory.
        test_dir (str): Path to the test data directory.
        img_height (int): Image height.
        img_width (int): Image width.
        batch_size (int): Batch size.

    Returns:
        tuple: Train, validation, and test DataLoader.
    """
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    train_size = int(0.8 * len(train_dataset))
    validation_size = len(train_dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [train_size, validation_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader


class LeNet5(nn.Module):
    """LeNet-5 model architecture."""

    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        """Forward pass."""
        x = torch.tanh(self.conv1(x))
        x = self.pool(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(model, train_loader, validation_loader, criterion, optimizer, num_epochs=5):
    """Train the model.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        validation_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        num_epochs (int): Number of training epochs.

    Returns:
        dict: Training and validation history.
    """
    train_history = {'accuracy': [], 'loss': []}
    val_history = {'accuracy': [], 'loss': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = correct / total
        train_loss = running_loss / len(train_loader)
        train_history['accuracy'].append(train_accuracy)
        train_history['loss'].append(train_loss)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        val_loss /= len(validation_loader)
        val_history['accuracy'].append(val_accuracy)
        val_history['loss'].append(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, '
              f'Validation Accuracy: {val_accuracy:.4f}')

    return train_history, val_history


def save_model(model, path):
    """Save the trained model to a file.

    Args:
        model (nn.Module): The trained model.
        path (str): Path to save the model file.
    """
    torch.save(model.state_dict(), path)


def save_training_history(history, path):
    """Save training history to a CSV file.

    Args:
        history (dict): Training history.
        path (str): Path to save the CSV file.
    """
    history_df = pd.DataFrame(history)
    history_df.to_csv(path, index=False)


def plot_training_history(history, path):
    """Plot training and validation accuracy.

    Args:
        history (dict): Training history.
        path (str): Path to save the plot.
    """
    plt.figure(figsize=(12, 8))
    plt.plot(history['epoch'], history['train_accuracy'], label='Train Accuracy')
    plt.plot(history['epoch'], history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.show()


def evaluate_model(model, test_loader):
    """Evaluate the model on the test dataset.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for test data.

    Returns:
        float: Test accuracy.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    print(f'Test Accuracy: {test_accuracy:.4f}')
    return test_accuracy


def main():
    """Main function to run the entire pipeline."""
    # Paths to directories
    train_dir = '/train'
    test_dir = '/test'

    # Hyperparameters
    img_height, img_width = 32, 32
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    model_path = 'lenet_pokemon_model.pth'
    history_path = 'training_history.csv'
    plot_path = 'training_validation_accuracy_plot.png'

    # Get data loaders
    train_loader, validation_loader, test_loader = get_data_loaders(
        train_dir, test_dir, img_height, img_width, batch_size
    )

    # Initialize model, loss function, and optimizer
    num_classes = len(train_loader.dataset.dataset.classes)
    model = LeNet5(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_history, val_history = train_model(
        model, train_loader, validation_loader, criterion, optimizer, num_epochs
    )

    # Save the model and training history
    save_model(model, model_path)
    history = {
        'epoch': range(1, num_epochs + 1),
        'train_accuracy': train_history['accuracy'],
        'train_loss': train_history['loss'],
        'val_accuracy': val_history['accuracy'],
        'val_loss': val_history['loss']
    }
    save_training_history(history, history_path)

    # Plot training history
    plot_training_history(history, plot_path)

    # Evaluate on test dataset
    evaluate_model(model, test_loader)


if __name__ == "__main__":
    main()
