from data_loader import data_loaders
import os


if __name__ == '__main__':
    data = data_loaders.Div2kDataLoader('train.csv', os.getcwd())
