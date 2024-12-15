import torch
from torch.optim import Adam
from torch.nn import MSELoss
from tqdm import tqdm

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cuda'):
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            optimizer.zero_grad()
            
            x = batch.x.to(device)  # [batch_size, seq_len, num_nodes, features]
            edge_index = batch.edge_index.to(device)
            edge_weight = batch.edge_attr.to(device)
            y = batch.y.to(device)  # [batch_size, num_nodes, num_nodes]
            
            out = model(x, edge_index, edge_weight)
            loss = criterion(out, y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.x.to(device)
                edge_index = batch.edge_index.to(device)
                edge_weight = batch.edge_attr.to(device)
                y = batch.y.to(device)
                
                out = model(x, edge_index, edge_weight)
                loss = criterion(out, y)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'best_model.pt')