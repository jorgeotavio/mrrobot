import torch
import numpy as np

# Carregar e preparar os dados
def load_and_prepare_data(filename):
    with open(filename, 'r') as file:
        content = file.read()
    
    rounds = content.split('&,&,&,&\n')
    
    X = []
    y = []
    for round_data in rounds:
        lines = round_data.strip().split('\n')
        for i in range(len(lines) - 1):
            current_state = list(map(float, lines[i].split(',')))
            next_state = list(map(float, lines[i + 1].split(',')))
            X.append(current_state)
            y.append(next_state[:2])
    
    X = np.array(X)
    y = np.array(y)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    return X_tensor, y_tensor
