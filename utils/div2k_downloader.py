'''
'''
import os
import requests
import wget
import csv
import zipfile
import cv2

if __name__ == '__main__':
    os.chdir('..')
    if 'data' not in os.listdir():
        os.mkdir('data')
    os.chdir('data')
    if 'div2k' not in os.listdir():
        os.mkdir('div2k')
    os.chdir('div2k')
    urls = ['http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip', \
            'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip', \
            'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip', \
            'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip']
    
    training_data = {}
    validation_data = {}
    hr_train = []
    lr_train = []
    hr_valid = []
    lr_valid = []

    for url in urls:
        file_name = url.split('/')[-1]
        folder_dir = file_name.split('.')[0]
        try:
            if (file_name) not in os.listdir() and (folder_dir not in os.listdir() and folder_dir[:-3] not in os.listdir()):
                print("[+] Downloading", file_name)
                file_name = wget.download(url)
        except Exception as e:
            print("[-] Error downloading file")
        if folder_dir not in os.listdir() and folder_dir[:-3] not in os.listdir():
            print("[+] Extracting to", folder_dir)
            try:
                with zipfile.ZipFile(file_name, 'r') as zip_:
                    zip_.extractall('.')
                print("[+] Extraction Complete")
            except Exception as e:
                print("[-] Error Extraction")
            print("[+]Cleaning up ", file_name)
            try:
                os.remove(file_name)
            except Exception as e:
                pass

    for folder in os.listdir():
        cwd = os.getcwd()
        if 'div2k' in folder.lower():
            if 'valid' in folder.lower():
                if 'lr' in folder.lower():
                    lr_valid = sorted(os.listdir(f'{folder}/X2'))
                    lr_valid = [f'{cwd}/{folder}/X2/{a}' for a in lr_valid]
                else:
                    hr_valid = sorted(os.listdir(f'{folder}'))
                    hr_valid = [f'{cwd}/{folder}/{a}' for a in hr_valid]
            elif 'train' in folder.lower():
                if 'lr' in folder.lower():
                    lr_train = sorted(os.listdir(f'{folder}/X2'))
                    lr_train = [f'{cwd}/{folder}/X2/{a}' for a in lr_train]
                else:
                    hr_train = sorted(os.listdir(f'{folder}'))
                    hr_train = [f'{cwd}/{folder}/{a}' for a in hr_train]


    hr = (2040,1356,3)
    lr = (1020, 678, 3)
    lr_valid = [a for a in lr_valid if cv2.imread(a).shape==lr]
    lr_train = [a for a in lr_train if cv2.imread(a).shape==lr]
    hr_train = [a for a in hr_train if cv2.imread(a).shape==hr]
    hr_valid = [a for a in hr_valid if cv2.imread(a).shape==hr]

    training_data = {**{"lr":"hr"}, **dict(zip(lr_train, hr_train))}
    valid_data = {**{"lr":"hr"}, **dict(zip(lr_valid, hr_valid))}
    with open('train.csv', 'w') as csv_:
        writer = csv.writer(csv_)
        for key, value in training_data.items():
            writer.writerow([key, value])
    
    with open('valid.csv', 'w') as csv_:
        writer = csv.writer(csv_)
        for key, value in valid_data.items():
            writer.writerow([key, value])
        
    