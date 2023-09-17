from base.base_data_loader import BaseDataLoader
from torchvision import datasets, transforms


class Div2kDataLoader(BaseDataLoader):
    def __init__(self):
        pass


    def __getitem__(self, idx):
        print("Get Item Implemented")