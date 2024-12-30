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
        
        # Output layer
        init_range = np.sqrt(6.0 / (node_num * gen_hid_num0 + node_num * node_num))
        self.output_weight = nn.Parameter(torch.FloatTensor(node_num * gen_hid_num0, node_num * node_num).uniform_(-init_range, init_range))
        self.output_bias = nn.Parameter(torch.zeros(node_num * node_num))
        
    def forward(self, noise_inputs, gcn_facts):
        # Process each time step through GCN
        gcn_outputs = []
        for t in range(self.window_size + 1):
            gcn_out = self.gcn_layers[t](gcn_facts[t], noise_inputs[t])
            gcn_out = self.dropout(gcn_out)  # Apply dropout after GCN
            gcn_outputs.append(gcn_out.view(1, -1))  # Reshape for LSTM
            
        # Stack GCN outputs for LSTM
        gcn_outputs = torch.stack(gcn_outputs, dim=1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(gcn_outputs)
        lstm_out = lstm_out[:, -1, :]  # Take last output
        lstm_out = self.dropout(lstm_out)  # Apply dropout after LSTM
        
        # Output layer
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

def get_wei_KL(adj_est, gnd):
    '''
    Function to calculate the edge weight KL divergence
    :param adj_est: prediction result
    :param gnd: ground-truth
    :return: edge weight KL divergence
    '''
    sum_est = 0
    for r in range(adj_est.shape[0]):
        for c in range(adj_est.shape[0]):
            sum_est += adj_est[r, c]
    sum_gnd = 0
    for r in range(gnd.shape[0]):
        for c in range(gnd.shape[0]):
            sum_gnd += gnd[r, c]
    p = gnd/sum_gnd
    q = adj_est/sum_est
    edge_wei_KL = 0
    for r in range(adj_est.shape[0]):
        for c in range(adj_est.shape[0]):
            cur_KL = 0
            if q[r, c]>0 and p[r, c]>0:
                cur_KL = p[r, c]*np.log(p[r, c]/q[r, c])
            edge_wei_KL += cur_KL

    return edge_wei_KL

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
    
    # Load data
    adjs = []
    for time_index in range(time_num):
        adjs.append(read_data(name_pre, time_index, node_num))
    max_elem = np.max(adjs)
    
    # Optimizers with same learning rates as TF version
    pre_gen_optimizer = optim.RMSprop(generator.parameters(), lr=0.005)  # Same as TF
    gen_optimizer = optim.RMSprop(generator.parameters(), lr=0.001)      # Same as TF
    disc_optimizer = optim.RMSprop(discriminator.parameters(), lr=0.001) # Same as TF
    
    # Training loop
    for t in range(window_size, time_num-3):
        # Prepare data
        gcn_facts = []
        for k in range(t-window_size, t+1):
            adj = adjs[k]/max_elem  # Normalize by max_elem like TF version
            gcn_fact = get_gcn_fact(adj, node_num)
            gcn_facts.append(torch.FloatTensor(gcn_fact).to(device))
            
        gnd = torch.FloatTensor(np.reshape(adjs[t+1], (1, node_num*node_num))/max_elem).to(device)
        
        # Pre-training generator with exact same early stopping as TF
        print("Pre-training generator...")
        loss_list = []
        for epoch in range(pre_epoch_num):
            noise_inputs = [torch.FloatTensor(gen_noise(node_num, node_num)).to(device) 
                          for _ in range(window_size + 1)]
            
            pre_gen_optimizer.zero_grad()
            gen_output = generator(noise_inputs, gcn_facts)
            pre_loss = torch.sum((gnd - gen_output) ** 2)
            
            # Add L2 regularization exactly like TF version
            l2_reg = 0
            for param in generator.parameters():
                l2_reg += decay_rate * torch.norm(param)
            pre_loss += l2_reg
            
            pre_loss.backward()
            pre_gen_optimizer.step()
            
            loss_list.append(pre_loss.item())
            if epoch % 100 == 0:
                print(f'Pre-Train #{epoch}, G-Loss: {pre_loss.item():.6f}')
                
            # Exact same early stopping as TF version
            if epoch > 500 and len(loss_list) > 2:
                if loss_list[-1] > loss_list[-2] and loss_list[-2] > loss_list[-3]:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Training GAN with same number of discriminator updates as TF
        print('Training GAN...')
        for epoch in range(epoch_num):
            # Train discriminator multiple times like TF
            for _ in range(n_critic):
                noise_inputs = [torch.FloatTensor(gen_noise(node_num, node_num)).to(device) 
                              for _ in range(window_size + 1)]
                
                disc_optimizer.zero_grad()
                gen_output = generator(noise_inputs, gcn_facts)
                disc_real, disc_logit_real = discriminator(gnd)
                disc_fake, disc_logit_fake = discriminator(gen_output.detach())
                
                # Wasserstein loss exactly like TF
                disc_loss = -(torch.mean(disc_logit_real) - torch.mean(disc_logit_fake))
                
                # Add L2 regularization
                l2_reg = 0
                for param in discriminator.parameters():
                    l2_reg += decay_rate * torch.norm(param)
                disc_loss += l2_reg
                
                disc_loss.backward()
                disc_optimizer.step()
                
                # Clip weights exactly like TF
                for p in discriminator.parameters():
                    p.data.clamp_(-clip_value, clip_value)
            
            # Train generator
            gen_optimizer.zero_grad()
            gen_output = generator(noise_inputs, gcn_facts)
            disc_fake, disc_logit_fake = discriminator(gen_output)
            
            # Generator loss exactly like TF
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
            adj = adjs[k]/max_elem
            gcn_fact = get_gcn_fact(adj, node_num)
            gcn_facts.append(torch.FloatTensor(gcn_fact).to(device))
            
        noise_inputs = [torch.FloatTensor(gen_noise(node_num, node_num)).to(device) 
                       for _ in range(window_size + 1)]
        
        # Set models to eval mode for prediction
        generator.eval()
        with torch.no_grad():
            output = generator(noise_inputs, gcn_facts)
            adj_est = output.cpu().numpy().reshape(node_num, node_num) * max_elem
            adj_est = (adj_est + adj_est.T) / 2  # Ensure symmetry
            np.fill_diagonal(adj_est, 0)  # No self-loops
            adj_est[adj_est < 0.01] = 0  # Same threshold as TF
        generator.train()  # Set back to training mode
            
        gnd = adjs[t+2]
        
        # Calculate metrics exactly like TF
        error = np.linalg.norm(gnd-adj_est, ord='fro')/(node_num*node_num)
        edge_wei_KL = get_wei_KL(adj_est, gnd)
        mis_rate = get_mis_rate(adj_est, gnd)
        
        # Print results
        print(f'#{t+2} Error: {error:.6f}')
        print(f'#{t+2} Edge Weight KL: {edge_wei_KL:.6f}')
        print(f'#{t+2} Mismatch Rate: {mis_rate:.6f}')
        
        print('Predicted adjacency matrix:')
        for r in range(node_num):
            for c in range(node_num):
                print(f'{adj_est[c, r]:.2f}', end=' ')
            print()
            
        print('Ground truth:')
        for r in range(node_num):
            for c in range(node_num):
                print(f'{gnd[c, r]:.2f}', end=' ')
            print()
        
        # Save results exactly like TF
        with open("+UCSB-LSTM_GAN_GCN(DrL2)-error.txt", 'a+') as f:
            f.write(f'{t+2} {error}\n')
            
        with open("+UCSB-LSTM_GAN_GCN(DrL2)-KL.txt", 'a+') as f:
            f.write(f'{t+2} {edge_wei_KL}\n')
            
        with open("+UCSB-LSTM_GAN_GCN(DrL2)-mis.txt", 'a+') as f:
            f.write(f'{t+2} {mis_rate}\n')

if __name__ == "__main__":
    main() 