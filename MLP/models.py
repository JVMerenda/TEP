import torch
import torch.nn as nn
from tqdm import tqdm
import logging

class PairLSTMClassifier(nn.Module):
    """
    LSTM that reads a time series of shape (seq_len, 2) => (valA_t, valB_t)
    for each step, and outputs a single binary label.
    """
    def __init__(self, lstm_hidden_dim=16, mlp_hidden_dim=32):
        super().__init__()
        
        # LSTM that takes input_dim=2
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # A small MLP to go from LSTM hidden_dim -> 1
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        x shape: (batch_size, seq_len, 2)
        Returns: (batch_size,) => probability that the two nodes are connected
        """
        # Pass through LSTM
        # lstm_out shape: (batch_size, seq_len, lstm_hidden_dim)
        # (h_n, c_n): each shape [num_layers, batch_size, hidden_dim]
        _, (h_n, _) = self.lstm(x)
        
        # Take final hidden state of last layer
        # shape: (batch_size, hidden_dim)
        h_final = h_n[-1]
        
        # Classify
        out = self.classifier(h_final)  # shape: (batch_size, 1)
        return out.squeeze(1)          # shape: (batch_size,)

def train_with_logging(model, train_loader, test_loader, criterion, optimizer, num_epochs=5):
    """
    model: the neural network model (e.g., PairLSTMClassifier)
    train_loader: DataLoader for training set
    test_loader:  DataLoader for test set
    criterion:    loss function, e.g., nn.BCELoss()
    optimizer:    e.g., torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs:   number of training epochs
    """

    # Set up basic logging (optional)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    logger = logging.getLogger(__name__)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Wrap the train_loader in tqdm
        train_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        
        for X_batch, y_batch in train_tqdm:
            # Forward
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate the loss
            running_loss += loss.item() * X_batch.size(0)
            
            # Show the current batch loss in the tqdm progress bar postfix
            train_tqdm.set_postfix({'batch_loss': loss.item()})
        
        # Compute average loss over the entire train_loader
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # -----------------------------
        # Evaluate on Test Set
        # -----------------------------
        model.eval()
        tp = fp = tn = fn = 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_pred = model(X_batch)
                predicted = (y_pred >= 0.5).float()
                
                p = predicted.int()
                t = y_batch.int()
                
                tp += torch.logical_and(p == 1, t == 1).sum().item()
                fp += torch.logical_and(p == 1, t == 0).sum().item()
                tn += torch.logical_and(p == 0, t == 0).sum().item()
                fn += torch.logical_and(p == 0, t == 1).sum().item()
        
        accuracy  = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Print final epoch metrics
        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_loss:.4f} | "
              f"Test Acc: {accuracy:.4f} | "
              f"Prec: {precision:.4f} | "
              f"Rec: {recall:.4f}")
        
        # Alternatively, use logging at INFO level:
        logger.info(f"Epoch={epoch+1}, Loss={epoch_loss:.4f}, "
                    f"Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

    