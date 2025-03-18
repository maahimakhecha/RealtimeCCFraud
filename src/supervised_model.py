# src/supervised_model.py

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CreditCardDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.X = self.data.drop('Class', axis=1).values.astype(np.float32)
        self.y = self.data['Class'].values.astype(np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SupervisedNet(nn.Module):
    def __init__(self, input_dim):
        super(SupervisedNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(16, 8)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(8, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        return x

def main():
    train_csv = os.path.join('data', 'processed', 'train.csv')
    test_csv = os.path.join('data', 'processed', 'test.csv')
    
    train_dataset = CreditCardDataset(train_csv)
    test_dataset = CreditCardDataset(test_csv)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    input_dim = train_dataset.X.shape[1]
    model = SupervisedNet(input_dim).to(device)
    
    criterion = nn.BCELoss()
    # Add weight decay to help with overfitting
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Compute class weights for imbalance
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_dataset.y), y=train_dataset.y)
    class_weights = {i: w for i, w in enumerate(weights)}
    print("Class weights:", class_weights)
    
    num_epochs = 20
    loss_history = []  # record average loss per epoch
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            # Apply sample weights
            sample_weights = torch.where(y_batch == 1, 
                                         torch.tensor(class_weights[1], dtype=torch.float32).to(device),
                                         torch.tensor(class_weights[0], dtype=torch.float32).to(device))
            loss = (loss * sample_weights).mean()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    # Plot Loss vs. Epoch and save as image
    model_dir = os.path.join('models')
    os.makedirs(model_dir, exist_ok=True)
    plt.figure(figsize=(8,6))
    plt.plot(range(1, num_epochs+1), loss_history, marker='o', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss vs. Epoch')
    plt.grid(True)
    loss_plot_path = os.path.join(model_dir, 'supervised_loss.png')
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Loss curve saved to {loss_plot_path}")
    
    # Evaluate on test set
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)
            outputs = model(X_batch)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    y_pred = (all_preds > 0.5).astype(int)
    auc_score = roc_auc_score(all_labels, all_preds)
    print(f"Test ROC AUC Score: {auc_score:.4f}")
    print("Classification Report:")
    print(classification_report(all_labels, y_pred))
    
    # Plot ROC Curve and save as image
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}', color='green')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Supervised Model')
    plt.legend(loc='lower right')
    roc_plot_path = os.path.join(model_dir, 'roc_curve.png')
    plt.savefig(roc_plot_path)
    plt.close()
    print(f"ROC curve saved to {roc_plot_path}")
    
    # Save the trained model weights
    model_path = os.path.join(model_dir, 'supervised_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Supervised model saved to {model_path}")

if __name__ == '__main__':
    main()
