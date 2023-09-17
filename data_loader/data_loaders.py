from base.base_data_loader import BaseDataLoader
from torchvision import datasets, transforms
from skimage import io


class Div2kDataLoader(BaseDataLoader):
    def __init__(self, csv_file, root_dir, transform=None):
        super().__init__(csv_file=csv_file, root_dir=root_dir, transform=transform)
        print("[+] Data Loaded with rows: ", super().__len__())


    def __getitem__(self, idx):
        lr_image_name = self.csv_dataframe.iloc[idx, 0]
        hr_image_name = self.csv_dataframe.iloc[idx, 1]
        lr_image = io.imread(lr_image_name)
        hr_image = io.imread(hr_image_name)
        sample = {'lr_image':lr_image, 'hr_image': hr_image}
        if self.transform:
            sample = self.transform(sample)
        
        return sample