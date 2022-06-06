import torch
import cnn_pytorch as cnn
from cnn_pytorch import Net

save_path = 'c:/Users/night/Documents/09/school/actual-masters/git/masters/models/saved_models/model_v1.pt'



# if __name__ == "__main__":
    # load model
model = torch.load(save_path)
model.eval()