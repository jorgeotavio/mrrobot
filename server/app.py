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

    torch.save(model.state_dict(), 'data/result.pth')
    return jsonify({'message': 'Treinada'})

@app.route('/validate', methods=['GET'])
def validate():
    X_val, y_val = load_and_prepare_data('enemies_data_validate.txt')
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)

    model.load_state_dict(torch.load('data/result.pth'))
    model.eval()

    total_loss = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)

    return jsonify({'average_loss': avg_loss})

@app.route('/predict', methods=['GET'])
def predict():
    model.load_state_dict(torch.load('data/result.pth'))
    model.eval()

    try:
        data = request.args
        enemyX = round(float(data.get('enemyX').replace(',', '.')), 2)
        enemyY = round(float(data.get('enemyY').replace(',', '.')), 2)
        myX = round(float(data.get('myX').replace(',', '.')), 2)
        myY = round(float(data.get('myY').replace(',', '.')), 2)
        inputs = torch.tensor([[enemyX, enemyY, myX, myY]], dtype=torch.float32)
        with torch.no_grad():
            prediction = model(inputs)
        return str(prediction[0][0].item())+","+str(prediction[0][1].item())
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
