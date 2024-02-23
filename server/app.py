import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from flask import Flask, request, jsonify
from data_manage import load_and_prepare_data

class SimpleRegressor(nn.Module):
    def __init__(self):
        super(SimpleRegressor, self).__init__()
        self.linear = nn.Linear(4, 2)  # 4 entradas para 2 saídas

    def forward(self, x):
        return self.linear(x)

app = Flask(__name__)
model = SimpleRegressor()

@app.route('/train', methods=['GET'])
def train():
    X, y = load_and_prepare_data('enemies_data.txt')
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(100):  # Número de épocas
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), 'simple_regressor.pth')
    return jsonify({'message': 'Model trained and saved'})

@app.route('/predict', methods=['GET'])
def predict():
    model.load_state_dict(torch.load('simple_regressor.pth'))
    model.eval()

    try:
        data = request.args
        enemyX = float(data.get('enemyX'))
        enemyY = float(data.get('enemyY'))
        myX = float(data.get('myX'))
        myY = float(data.get('myY'))
        inputs = torch.tensor([[enemyX, enemyY, myX, myY]], dtype=torch.float32)
        with torch.no_grad():
            prediction = model(inputs)
        return jsonify({'nextEnemyX': prediction[0][0].item(), 'nextEnemyY': prediction[0][1].item()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
