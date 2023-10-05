import os
import base64
import gzip
import binascii
import struct
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from io import BytesIO
from torch.utils.data import Dataset, DataLoader

DATASET_PATH = "D:\\Documents\\_Programming\\C# Projects\\Chess-Challenge\\Chess-Challenge\\src\\My Bot\\training\\chessData-Proccessed.csv"
MODEL_FOLDER = "D:\\Documents\\_Programming\\C# Projects\\Chess-Challenge\\Chess-Challenge\\src\\My Bot\\Models\\"
STRING_MODEL_PATH = "D:\\Documents\\_Programming\\C# Projects\\Chess-Challenge\\Chess-Challenge\\src\\My Bot\\Models\\string_model.txt"
MODEL_PREFIX = "torch_model"
# ROW_LIMIT = 750000
ROW_LIMIT = 10


# Step 1: Preprocess dataset
def preprocess_data(filename, row_limit=None):
    data = pd.read_csv(filename, nrows=row_limit)
    data_size = data.shape[0]
    inputs = [[0.0]*776] * data_size
    evaluations = [None] * data_size

    for index, row in tqdm(data.iterrows(), total=data_size):
        # decode base64
        decoded = base64.b64decode(row['Input'])
        # decompress GZip
        with gzip.GzipFile(fileobj=BytesIO(decoded)) as f:
            byte_array = f.read()
        # Convert byte array to float array
        float_array = [struct.unpack('f', byte_array[i:i+4])[0] for i in range(0, len(byte_array), 4)]
        inputs[index] = float_array
        evaluations[index] = float(row['Evaluation'])

    return inputs, evaluations


# Custom Dataset class
class ChessDataset(Dataset):
    def __init__(self, inputs, evaluations):
        self.inputs = inputs
        self.evaluations = evaluations

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.inputs[idx]), self.evaluations[idx]


# Step 2: Create the neural network
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(776, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out


# Step 3: Train the neural network
def train_neural_network(dataset_file_path, save_folder, save_prefix, save_interval = 10, batch_size=64, num_epochs=128, learn_rate=0.002, print_interval=10240):
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Preprocessing Data...")
    inputs, evaluations = preprocess_data(dataset_file_path, ROW_LIMIT)
    dataset = ChessDataset(inputs, evaluations)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    print(f"Moving data to {device}...")
    model = NeuralNet().to(device)  # Move model to GPU
    
    print(f"Loading model {save_prefix}...")
    latest_epoch = load_latest_model(model, save_folder, save_prefix) or 0  # Load the latest model
    print(f"Latest epoch: {latest_epoch}")
    
    print(f"Hex string len: {len(convert_weights_biases_to_hex_string(model))}")
    print(f"Base 64 len: {len(convert_weights_biases_to_base64_string(model))}")
    print(f"String len: {len(convert_weights_biases_to_string(model))}")
    
    return
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learn_rate)

    for epoch in range(latest_epoch, latest_epoch+num_epochs):  # Start from the latest epoch
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.view(-1, 1).float().to(device)  # Move inputs and labels to GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if (i+1) % print_interval == 0:
                print(f'Epoch [{epoch+1}/{latest_epoch+num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

                
        # Save the model every 'save_interval' epochs
        if (epoch + 1) % save_interval == 0:
            save_model(model, save_folder, save_prefix, epoch+1)


def save_model(model, folder, file_prefix, epoch):
    file_name = f"{file_prefix}_epoch_{epoch}.pth"
    file_path = os.path.join(folder, file_name)
    print(f"Saving model to {file_path}")
    torch.save(model.state_dict(), file_path)


def load_latest_model(model, folder, file_prefix):
    # Get all the files in the folder with the file_prefix
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.startswith(file_prefix)]
    if not files:
        return  # If no files found, do nothing

    # Get the latest epoch number
    latest_epoch = max(int(f.split('_')[-1].split('.')[0]) for f in files)

    # Load the model weights from the latest epoch
    latest_file = os.path.join(folder, f"{file_prefix}_epoch_{latest_epoch}.pth")
    model.load_state_dict(torch.load(latest_file))
    return latest_epoch  # Return the latest epoch number


def convert_weights_biases_to_hex_string(model):
    model = model.to("cpu")
    weights_biases = [
        model.fc1.weight.data.numpy().flatten(),
        model.fc1.bias.data.numpy().flatten(),
        model.fc2.weight.data.numpy().flatten(),
        model.fc2.bias.data.numpy().flatten(),
        model.fc3.weight.data.numpy().flatten(),
        model.fc3.bias.data.numpy().flatten(),
    ]
    wb_hex = ';'.join(''.join(binascii.hexlify(struct.pack('f', x)).decode('utf-8') for x in wb) for wb in weights_biases)
    with open(STRING_MODEL_PATH, "w") as file:
        file.write(wb_hex)
    return wb_hex

def convert_weights_biases_to_base64_string(model):
    model = model.to("cpu")
    weights_biases = [
        model.fc1.weight.data.numpy().flatten(),
        model.fc1.bias.data.numpy().flatten(),
        model.fc2.weight.data.numpy().flatten(),
        model.fc2.bias.data.numpy().flatten(),
        model.fc3.weight.data.numpy().flatten(),
        model.fc3.bias.data.numpy().flatten(),
    ]
    wb_str = ';'.join(','.join(map(str, wb.flatten())) for wb in weights_biases)
    wb_bytes = wb_str.encode('utf-8')  # Convert string to bytes
    wb_base64 = base64.b64encode(wb_bytes)  # Encode bytes to base64
    return wb_base64.decode('utf-8')  # Convert base64 bytes back to string


def convert_weights_biases_to_string(model):
    model = model.to("cpu")
    # Assuming `model` is an instance of the NeuralNet class
    # We're also assuming that the weights and biases are stored as numpy arrays in the model
    weights_biases = [
        model.fc1.weight.data.numpy().flatten(),
        model.fc1.bias.data.numpy().flatten(),
        model.fc2.weight.data.numpy().flatten(),
        model.fc2.bias.data.numpy().flatten(),
        model.fc3.weight.data.numpy().flatten(),
        model.fc3.bias.data.numpy().flatten(),
    ]
    # Convert all values to strings and join them
    wb_str = ';'.join(','.join(map(str, wb.flatten())) for wb in weights_biases)
    return wb_str


# Example usage
train_neural_network(DATASET_PATH, MODEL_FOLDER, MODEL_PREFIX, save_interval=4)
