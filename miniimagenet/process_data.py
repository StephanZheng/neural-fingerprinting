__author__ = 'user'
__author__ = 'user'
import os
import shutil
import tqdm
import argparse

parser = argparse.ArgumentParser(description='Process data')
parser.add_argument('--directory',type=str)
args = parser.parse_args()


def process_class(im_class_path):
    list_images = []
    for images in os.listdir(im_class_path):
        list_images.append(images)



count_class = 0
for dirs in os.listdir(args.directory):

    count_class = count_class + 1
    if(count_class >=2):
        break


