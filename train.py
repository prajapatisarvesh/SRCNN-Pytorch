from data_loader import data_loaders


if __name__ == '__main__':
    data = data_loaders.Div2kDataLoader()
    data.__getitem__(1)