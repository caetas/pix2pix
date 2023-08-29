from config import data_raw_dir
from data.Dataloaders import create_train_loader, create_test_loader
import matplotlib.pyplot as plt
from models.CNET import train_cnet

train_loader = create_train_loader(data_raw_dir, 64)
test_loader = create_test_loader(data_raw_dir, 64)

train_cnet(train_loader, test_loader)