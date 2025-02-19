import copy
import torch
import torch.nn as nn
from tqdm import tqdm
import logging
from dataset import *
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR

class SimplePairClassifier(nn.Module):
    """
    A simpler classifier that uses CNN instead of LSTM
    """
    def __init__(self, input_channels=2, seq_length=40):
        super().__init__()
        
        # 1D CNN layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )
        
        # MLP layers
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # x shape: (batch_size, 2, seq_length)
        x = self.conv_layers(x)  # Shape: (batch_size, 64, 1)
        x = x.squeeze(-1)  # Shape: (batch_size, 64)
        x = self.classifier(x)  # Shape: (batch_size, 1)
        return x.squeeze(-1)

class PairLSTMClassifier(nn.Module):
    def __init__(self, lstm_hidden_dim=64, mlp_hidden_dim=32):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=lstm_hidden_dim,
            num_layers=1,  # Reduce layers
            batch_first=True,
            bidirectional=False  # Start with unidirectional
        )
        
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(lstm_hidden_dim),
            nn.Dropout(0.3),  # Add dropout
            nn.Linear(lstm_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),  # Add dropout
            nn.Linear(mlp_hidden_dim, 1)
        )
    
    def forward(self, x):
        lstm_out, (h_n, _) = self.lstm(x)
        
        # Use last hidden state
        h_final = h_n[-1]  # shape: (batch_size, hidden_dim)
        
        # Classify
        out = self.classifier(h_final)
        return out.squeeze(1)
    

def train_with_logging_and_cm(
    model: nn.Module,
    train_loader: torch.utils.data.Dataset,
    test_loader: torch.utils.data.Dataset,
    criterion,
    optimizer,
    num_epochs=5,
    batch_size=16,
    threshold=0.5,
    device='cpu'
):
    """
    Trains 'model' on a balanced version of 'train_dataset', 
    evaluates each epoch on 'test_dataset', 
    and prints a confusion matrix at the end.
    """
    
    # Optional logging setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    logger = logging.getLogger(__name__)

    # Learning rate scheduler
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
)
    
    best_f1 = 0
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # TQDM progress bar for this epoch
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        
        for X_batch, y_batch in train_pbar:
            # X_batch shape: (batch_size, seq_len, 2)
            # y_batch shape: (batch_size,)
            y_pred = model(X_batch.to(device).permute(0,2,1))
            loss = criterion(y_pred, y_batch.to(device).squeeze().float()).cpu()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)
            
            # Update the TQDM progress bar
            train_pbar.set_postfix({'batch_loss': loss.item()})
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Evaluate on test set
        model.eval()
        tp = fp = tn = fn = 0
        val_losses = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_out = model(X_batch.to(device).permute(0,2,1))
                val_loss = criterion(y_out, y_batch.to(device).squeeze().float()).cpu()
                val_losses.append(val_loss.item())
                
                predicted = (torch.sigmoid(y_out)).float().cpu()
                p = predicted.int()
                t = y_batch.int()
                tp += torch.logical_and(p == 1, t == 1).sum().item()
                fp += torch.logical_and(p == 1, t == 0).sum().item()
                tn += torch.logical_and(p == 0, t == 0).sum().item()
                fn += torch.logical_and(p == 0, t == 1).sum().item()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn)>0 else 0
        precision = tp / (tp + fp) if (tp + fp)>0 else 0
        recall = tp / (tp + fn) if (tp + fn)>0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Update scheduler with validation loss
        avg_val_loss = np.mean(val_losses)
        scheduler.step(avg_val_loss)
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            best_model = copy.deepcopy(model)
        
        # Print or log epoch metrics
        epoch_msg = (f"[Epoch {epoch+1}/{num_epochs}] "
                     f"Train Loss: {epoch_loss:.4f} | "
                     f"Val Loss: {avg_val_loss:.4f} | "
                     f"Test Acc: {accuracy:.4f} | "
                     f"Prec: {precision:.4f} | "
                     f"Rec: {recall:.4f} | "
                     f"F1: {f1:.4f}")
        print(epoch_msg)
        logger.info(epoch_msg)
    
    # Final evaluation and confusion matrix
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_out = model(X_batch.to(device).permute(0,2,1))
            predicted = (torch.sigmoid(y_out) > threshold).cpu().numpy()
            all_predictions.extend(predicted.flatten())
            all_targets.extend(y_batch.numpy().flatten())
    
    # Calculate final metrics using sklearn
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    print("\nClassification Report:")
    print(classification_report(all_targets, all_predictions))
        
class EnhancedPairLSTMClassifier(nn.Module):
    def __init__(self, lstm_hidden_dim=768, mlp_hidden_dim=32, dropout=0.3):
        super().__init__()
        
        self.lstm_hidden_dim = lstm_hidden_dim
        
        # Initial feature extraction
        self.feature_proj = nn.Linear(2, lstm_hidden_dim//2)
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_dim//2,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Self-attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden_dim*2,
            num_heads=8,
            dropout=dropout
        )
        
        # Classifier with residual connections
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_hidden_dim*2),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim*2, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim//2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim//2, 1)
        )
        
    def forward(self, x):
        # Initial feature projection
        x = self.feature_proj(x)
        
        # LSTM processing
        lstm_out, (h_n, _) = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1)
        )
        attn_out = attn_out.transpose(0, 1)
        
        # Global max pooling and average pooling
        max_pool = torch.max(attn_out, dim=1)[0]
        avg_pool = torch.mean(attn_out, dim=1)
        
        # Combine features
        combined = max_pool + avg_pool
        
        # Classification
        out = self.classifier(combined)
        return out.squeeze(1)
    
def train_with_logging_and_cm2(
    model: nn.Module,
    train_loader: torch.utils.data.Dataset,
    test_loader: torch.utils.data.Dataset,
    criterion,
    optimizer,
    num_epochs=5,
    batch_size=16,
    device='cpu'
):
    """
    Trains 'model' on a balanced version of 'train_dataset', 
    evaluates each epoch on 'test_dataset', 
    and prints a confusion matrix at the end.
    """
    
    # Optional logging setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    logger = logging.getLogger(__name__)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=optimizer.param_groups[0]['lr'],
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )
    
    best_val_loss = float('inf')
    best_model = None
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # TQDM progress bar for this epoch
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        
        for X_batch, y_batch in train_pbar:
            # X_batch shape: (batch_size, seq_len, 2)
            # y_batch shape: (batch_size,)
            y_pred = model(X_batch.to(device))
            loss = criterion(y_pred, y_batch.to(device).squeeze().float())
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item() * X_batch.size(0)
            
            # Update the TQDM progress bar
            train_pbar.set_postfix({'batch_loss': loss.item()})
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Evaluate on test set
        model.eval()
        val_losses = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_out = model(X_batch.to(device))
                val_loss = criterion(y_out, y_batch.to(device).squeeze().float())
                val_losses.append(val_loss.item())
                
                predicted = (torch.sigmoid(y_out) > 0.5).cpu().numpy()
                all_predictions.extend(predicted.flatten())
                all_targets.extend(y_batch.numpy().flatten())
        
        # Calculate metrics
        avg_val_loss = np.mean(val_losses)
        cm = confusion_matrix(all_targets, all_predictions)
        tp = cm[1,1]
        fp = cm[0,1]
        tn = cm[0,0]
        fn = cm[1,0]
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn)>0 else 0
        precision = tp / (tp + fp) if (tp + fp)>0 else 0
        recall = tp / (tp + fn) if (tp + fn)>0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = copy.deepcopy(model)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\nEarly stopping triggered")
                break
        
        # Print or log epoch metrics
        epoch_msg = (f"[Epoch {epoch+1}/{num_epochs}] "
                     f"Train Loss: {epoch_loss:.4f} | "
                     f"Val Loss: {avg_val_loss:.4f} | "
                     f"Test Acc: {accuracy:.4f} | "
                     f"Prec: {precision:.4f} | "
                     f"Rec: {recall:.4f} | "
                     f"F1: {f1:.4f}")
        print(epoch_msg)
        logger.info(epoch_msg)

    # Restore best model
    model.load_state_dict(best_model.state_dict())
    
    # Final evaluation and confusion matrix
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_out = model(X_batch.to(device))
            predicted = (torch.sigmoid(y_out) > 0.5).cpu().numpy()
            all_predictions.extend(predicted.flatten())
            all_targets.extend(y_batch.numpy().flatten())
    
    # Plot final confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    print("\nClassification Report:")
    print(classification_report(all_targets, all_predictions))
    
    return model