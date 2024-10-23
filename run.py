import torch
from torch.utils.data import Dataset, DataLoader
import json
import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'

class SiameseDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input1 = self.data[idx][0]
        input2 = self.data[idx][1]
        return (input1, input2)


# hyperparameter
batch_size = 1  
distance_metric = 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pretrained model
model_save_path ='pre_trained/6RIS.pth'


# Test data 
filename = 'data_demo.json'
with open(filename, "r") as fp:
    test_data = json.load(fp)
test_data = [(torch.Tensor(input1), torch.Tensor(input2)) for input1, input2 in test_data]
test_dataset = SiameseDataset(test_data)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Load Model 
test_model = torch.load(model_save_path)
test_model.eval()
with torch.no_grad():
    for step, data in enumerate(test_dataloader):
        input1, input2 = data
        input1, input2 = input1.to(device), input2.to(device)
        emd1,emd2 = test_model(input1, input2)
        distances = torch.norm(emd1 - emd2, p=2, dim=1)
        condition = distances < distance_metric
        output = torch.zeros(distances.size())
        output[condition] = 1
        output = output.int().tolist()
        if output[0] == 0:
            print("These two IPv6 addresses are not correlated")
        else:
            print("These two IPv6 addresses not correlated")