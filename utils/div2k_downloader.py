import os
import requests
import wget
import pandas as pd


if __name__ == '__main__':
    os.chdir('..')
    if 'data' not in os.listdir():
        os.mkdir('data')
    os.chdir('data')
    urls = ['http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip', \
            'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip', \
            'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip', \
            'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip']
    
    for url in urls:
        file_name = wget(urls)
        