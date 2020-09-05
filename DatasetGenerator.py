import numpy as np
import torch


class DatasetGenerator:
    
    def __init__(self, generator, seq_length=40, noise_dim=100, dataset=None):
        
        self.generator = generator
        self.seq_length = seq_length
        self.noise_dim = noise_dim
        self.dataset = dataset
        
        
    def generate(self, output_name=None, batch_size=4, deltas_list=None, size=1000):
        
        if deltas_list:
            noise = torch.randn(len(deltas_list), self.seq_length, self.noise_dim)
            deltas = torch.FloatTensor(deltas_list).view(-1, 1, 1).repeat(1, self.seq_length, 1)
            
            if self.dataset:
                deltas = self.dataset.normalize_deltas(deltas)
                
            noise = torch.cat((noise, deltas), dim=2)
            
        else:
            noise = torch.randn(size, self.seq_length, self.noise_dim)
            
        output_list = []
        for batch in noise.split(batch_size):
            output_list.append(self.generator(batch))
            
        output_tensor = torch.cat(output_list, dim=0)
        
        if self.dataset:
            output_tensor = self.dataset.denormalize(output_tensor)
            
        if output_name:
            np.save(output_name, output_tensor.detach().numpy())
        else:
            return output_tensor
        
