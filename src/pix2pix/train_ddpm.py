from config import data_raw_dir
from data.Dataloaders import create_train_loader, create_test_loader
import matplotlib.pyplot as plt
from models.DDPM import train_DDPM

train_loader = create_train_loader(data_raw_dir, 80)
test_loader = create_test_loader(data_raw_dir, 80)

train_DDPM(train_loader, test_loader)