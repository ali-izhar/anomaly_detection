import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

class GCNLayer(nn.Module):
    def __init__(self, node_num, hid_num):
        super(GCNLayer, self).__init__()
        init_range = np.sqrt(6.0 / (node_num + hid_num))
        self.weight = nn.Parameter(torch.FloatTensor(node_num, hid_num).uniform_(-init_range, init_range))
        
    def forward(self, gcn_fact, noise):
        # GCN operation: gcn_fact @ noise @ weight
        gcn_conv = torch.matmul(gcn_fact, noise)
        return torch.sigmoid(torch.matmul(gcn_conv, self.weight))

class Generator(nn.Module):
    def __init__(self, node_num, window_size, gen_hid_num0, dropout_rate=0.0):
        super(Generator, self).__init__()
        self.node_num = node_num
        self.window_size = window_size
        self.gen_hid_num0 = gen_hid_num0
        self.dropout = nn.Dropout(dropout_rate)
        
        # GCN layers
        self.gcn_layers = nn.ModuleList([
            GCNLayer(node_num, gen_hid_num0) for _ in range(window_size + 1)
        ])
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=node_num * gen_hid_num0,
            hidden_size=node_num * gen_hid_num0,
            num_layers=1,
            batch_first=True
        )
        
        # Output layer - modified for binary prediction
        init_range = np.sqrt(6.0 / (node_num * gen_hid_num0 + node_num * node_num))
        self.output_weight = nn.Parameter(torch.FloatTensor(node_num * gen_hid_num0, node_num * node_num).uniform_(-init_range, init_range))
        self.output_bias = nn.Parameter(torch.zeros(node_num * node_num))
        
    def forward(self, noise_inputs, gcn_facts):
        # Process each time step through GCN
        gcn_outputs = []
        for t in range(self.window_size + 1):
            gcn_out = self.gcn_layers[t](gcn_facts[t], noise_inputs[t])
            gcn_out = self.dropout(gcn_out)  # Apply dropout
            gcn_outputs.append(gcn_out.view(1, -1))  # Reshape for LSTM
            
        # Stack GCN outputs for LSTM
        gcn_outputs = torch.stack(gcn_outputs, dim=1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(gcn_outputs)
        lstm_out = lstm_out[:, -1, :]  # Take last output
        lstm_out = self.dropout(lstm_out)  # Apply dropout
        
        # Output layer with sigmoid for binary prediction
        gen_output = torch.sigmoid(torch.matmul(lstm_out, self.output_weight) + self.output_bias)
        return gen_output

class Discriminator(nn.Module):
    def __init__(self, node_num, disc_hid_num, dropout_rate=0.0):
        super(Discriminator, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Input layer -> Hidden layer
        init_range1 = np.sqrt(6.0 / (node_num * node_num + disc_hid_num))
        self.weight1 = nn.Parameter(torch.FloatTensor(node_num * node_num, disc_hid_num).uniform_(-init_range1, init_range1))
        self.bias1 = nn.Parameter(torch.zeros(disc_hid_num))
        
        # Hidden layer -> Output layer
        init_range2 = np.sqrt(6.0 / (disc_hid_num + 1))
        self.weight2 = nn.Parameter(torch.FloatTensor(disc_hid_num, 1).uniform_(-init_range2, init_range2))
        self.bias2 = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        x = self.dropout(x)  # Apply dropout to input
        h1 = torch.sigmoid(torch.matmul(x, self.weight1) + self.bias1)
        h1 = self.dropout(h1)  # Apply dropout to hidden layer
        logit = torch.matmul(h1, self.weight2) + self.bias2
        output = torch.sigmoid(logit)
        return output, logit

def read_data(name_pre, time_index, node_num):
    '''
    Function to read binary edge list file
    '''
    print('Read network snapshot #%d' % (time_index))
    curAdj = np.mat(np.zeros((node_num, node_num)))
    f = open('%s_%d.txt' % (name_pre, time_index))
    line = f.readline()
    while line:
        seq = line.split()
        src = int(seq[0])
        tar = int(seq[1])
        curAdj[src, tar] = 1  # Binary edges
        curAdj[tar, src] = 1  # Symmetric
        line = f.readline()
    f.close()
    return curAdj

def gen_noise(m, n):
    return np.random.uniform(0, 1., size=[m, n])

def get_gcn_fact(adj, node_num):
    adj_ = adj + np.eye(node_num, node_num)
    row_sum = np.array(adj_.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.mat(np.diag(d_inv_sqrt))
    return d_mat_inv_sqrt * adj_ * d_mat_inv_sqrt

def get_binary_accuracy(adj_est, gnd):
    """Calculate accuracy for binary edge prediction."""
    predictions = (adj_est > 0.5).astype(np.float32)
    correct = (predictions == gnd).sum()
    total = gnd.size
    return correct / total

def get_mis_rate(adj_est, gnd):
    """Calculate mismatch rate for binary edge prediction."""
    node_num = adj_est.shape[0]
    predictions = (adj_est > 0.5).astype(np.float32)
    mis_sum = ((predictions != gnd).sum())
    return mis_sum / (node_num * node_num)

def main():
    # Model parameters aligned with paper
    node_num = 38
    time_num = 1000        # Reduced timesteps
    window_size = 10       # Paper suggests moderate window size
    name_pre = "./data/SBM/edge_list"
    
    # Architecture parameters
    dropout_rate = 0.1     # Keep moderate dropout
    decay_rate = 5e-5      # Reduced L2 regularization
    pre_epoch_num = 2000   # Increased pre-training
    epoch_num = 4000       # Keep same training length
    gen_hid_num0 = 32      # Keep GCN capacity
    gen_hid_num1 = 128     # Keep LSTM capacity
    disc_hid_num = 256     # Keep discriminator capacity
    
    # Learning rates
    pre_gen_lr = 0.001     # Increased for better pre-training
    gen_lr = 0.0002        # Increased slightly
    disc_lr = 0.0001       # Keep discriminator slower
    
    # Training parameters
    n_critic = 5           # Train discriminator more
    clip_value = 0.01      # Weight clipping value
    
    # Use CUDA with RTX 4090
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    
    print("\nTraining Configuration:")
    print(f"- Time steps: {time_num}")
    print(f"- Window size: {window_size}")
    print(f"- Pre-training epochs: {pre_epoch_num}")
    print(f"- GAN training epochs: {epoch_num}")
    print(f"- Generator hidden dims: {gen_hid_num0}, {gen_hid_num1}")
    print(f"- Discriminator hidden dims: {disc_hid_num}")
    print(f"- Dropout rate: {dropout_rate}")
    print(f"- L2 decay rate: {decay_rate}")
    print(f"- Learning rates - Pre-gen: {pre_gen_lr}, Gen: {gen_lr}, Disc: {disc_lr}")
    print(f"- Critic iterations: {n_critic}")
    print()
    
    # Create models directory if it doesn't exist
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Initialize models with dropout
    generator = Generator(node_num, window_size, gen_hid_num0, dropout_rate).to(device)
    discriminator = Discriminator(node_num, disc_hid_num, dropout_rate).to(device)
    
    # Binary cross entropy loss for pre-training
    bce_loss = nn.BCELoss()
    
    # Optimizers
    pre_gen_optimizer = optim.Adam(generator.parameters(), lr=pre_gen_lr, betas=(0.5, 0.999))
    gen_optimizer = optim.Adam(generator.parameters(), lr=gen_lr, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=disc_lr, betas=(0.5, 0.999))
    
    # Learning rate schedulers with longer patience
    pre_gen_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        pre_gen_optimizer, mode='min', factor=0.5, patience=200, verbose=True)
    gen_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        gen_optimizer, mode='min', factor=0.5, patience=300, verbose=True)
    disc_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        disc_optimizer, mode='min', factor=0.5, patience=300, verbose=True)
    
    # Training loop
    avg_accuracy = 0.0
    avg_mis = 0.0
    cal_count = 0
    
    for t in range(window_size, time_num-2):
        # Prepare data
        gcn_facts = []
        for k in range(t-window_size, t+1):
            adj = read_data(name_pre, k, node_num)
            gcn_fact = get_gcn_fact(adj, node_num)
            gcn_facts.append(torch.FloatTensor(gcn_fact).to(device))
            
        gnd = torch.FloatTensor(np.reshape(read_data(name_pre, t+1, node_num), 
                                         (1, node_num*node_num))).to(device)
        
        # Pre-training generator with binary cross entropy loss and L2 regularization
        print("Pre-training generator...")
        best_pre_loss = float('inf')
        no_improve_count = 0
        for epoch in range(pre_epoch_num):
            noise_inputs = [torch.FloatTensor(gen_noise(node_num, node_num)).to(device) 
                          for _ in range(window_size + 1)]
            
            pre_gen_optimizer.zero_grad()
            gen_output = generator(noise_inputs, gcn_facts)
            pre_loss = bce_loss(gen_output, gnd)
            
            # Reduced L2 regularization during pre-training
            l2_reg = 0
            for param in generator.parameters():
                l2_reg += decay_rate * 0.1 * torch.norm(param)
            pre_loss += l2_reg
            
            pre_loss.backward()
            pre_gen_optimizer.step()
            
            # Update learning rate and early stopping
            if pre_loss.item() < best_pre_loss:
                best_pre_loss = pre_loss.item()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            pre_gen_scheduler.step(pre_loss)
            
            if epoch % 100 == 0:
                print(f'Pre-Train #{epoch}, G-Loss: {pre_loss.item():.6f}')
            
            # Early stopping with longer patience
            if no_improve_count > 300:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Training GAN
        print('Train the GAN')
        for epoch in range(epoch_num):
            # Train discriminator multiple times
            for _ in range(n_critic):
                noise_inputs = [torch.FloatTensor(gen_noise(node_num, node_num)).to(device) 
                              for _ in range(window_size + 1)]
                
                disc_optimizer.zero_grad()
                gen_output = generator(noise_inputs, gcn_facts)
                disc_real, disc_logit_real = discriminator(gnd)
                disc_fake, disc_logit_fake = discriminator(gen_output.detach())
                
                # Wasserstein loss
                disc_loss = -(torch.mean(disc_logit_real) - torch.mean(disc_logit_fake))
                
                # Add L2 regularization
                l2_reg = 0
                for param in discriminator.parameters():
                    l2_reg += decay_rate * torch.norm(param)
                disc_loss += l2_reg
                
                disc_loss.backward()
                disc_optimizer.step()
                
                # Clip weights
                for p in discriminator.parameters():
                    p.data.clamp_(-clip_value, clip_value)
            
            # Train generator
            gen_optimizer.zero_grad()
            gen_output = generator(noise_inputs, gcn_facts)
            disc_fake, disc_logit_fake = discriminator(gen_output)
            
            # Generator tries to maximize discriminator output on fake data
            gen_loss = -torch.mean(disc_logit_fake)
            
            # Add L2 regularization
            l2_reg = 0
            for param in generator.parameters():
                l2_reg += decay_rate * torch.norm(param)
            gen_loss += l2_reg
            
            gen_loss.backward()
            gen_optimizer.step()
            
            if epoch % 100 == 0:
                print(f'GAN-Train #{epoch}, D-Loss: {disc_loss.item():.6f}, G-Loss: {gen_loss.item():.6f}')
        
        # Prediction
        print("Making prediction...")
        gcn_facts = []
        for k in range(t-window_size+1, t+2):
            adj = read_data(name_pre, k, node_num)
            gcn_fact = get_gcn_fact(adj, node_num)
            gcn_facts.append(torch.FloatTensor(gcn_fact).to(device))
            
        noise_inputs = [torch.FloatTensor(gen_noise(node_num, node_num)).to(device) 
                       for _ in range(window_size + 1)]
        
        # Set models to eval mode for prediction
        generator.eval()
        with torch.no_grad():
            output = generator(noise_inputs, gcn_facts)
            adj_est = output.cpu().numpy().reshape(node_num, node_num)
            adj_est = (adj_est + adj_est.T) / 2  # Ensure symmetry
            np.fill_diagonal(adj_est, 0)  # No self-loops
        generator.train()  # Set back to training mode
            
        gnd = read_data(name_pre, t+2, node_num)
        
        # Calculate metrics
        accuracy = get_binary_accuracy(adj_est, gnd)
        mis_rate = get_mis_rate(adj_est, gnd)
        avg_accuracy += accuracy
        avg_mis += mis_rate
        cal_count += 1
        
        # Print results
        print(f'Accuracy: {accuracy:.4f}, Mismatch Rate: {mis_rate:.4f}')
        print('Predicted adjacency matrix:')
        for r in range(node_num):
            for c in range(node_num):
                print(f'{int(adj_est[c, r] > 0.5)}', end=' ')  # Print binary predictions
            print()
            
        print('Ground truth:')
        for r in range(node_num):
            for c in range(node_num):
                print(f'{int(gnd[c, r])}', end=' ')
            print()
    
    # Print average metrics
    if cal_count > 0:
        print(f'\nAverage Accuracy: {avg_accuracy/cal_count:.4f}')
        print(f'Average Mismatch Rate: {avg_mis/cal_count:.4f}')
    
    # After training loop ends, save the models
    print("\nSaving models...")
    model_state = {
        'generator_state': generator.state_dict(),
        'discriminator_state': discriminator.state_dict(),
        'model_params': {
            'node_num': node_num,
            'window_size': window_size,
            'gen_hid_num0': gen_hid_num0,
            'disc_hid_num': disc_hid_num,
            'dropout_rate': dropout_rate,
            'decay_rate': decay_rate
        }
    }
    torch.save(model_state, models_dir / 'lstm_gan_gcn_drl2_model.pt')
    print("Models saved successfully!")

if __name__ == "__main__":
    main() 