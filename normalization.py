from reinhard import *
import os
import glob
import cv2
import numpy as np

INPUT_DIR = "./Test_4cl_amostra/3/"
REFERENCE_GRID = "./matrix.png"

def create_output_dirs(base_dirs):
    
    normalized_dir = f"{base_dirs}_normalized"

    if not os.path.exists(normalized_dir):
        os.makedirs(normalized_dir)

def normalize_images(input_dir, target_image_path):
    
    ''' Load our matrix and calculates statistics'''
    target_image = cv2.imread(target_image_path)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
    target_lab = rgb_to_lab(target_image)
    l, a, b = cv2.split(target_lab)
    target_mean = np.array([np.mean(l), np.mean(a), np.mean(b)])
    target_std = np.array([np.std(l), np.std(a), np.std(b)])

    create_output_dirs(input_dir)


    image_paths = glob.glob(f"{input_dir}/*.png")
    output_dir = f"{input_dir}_normalized"
    

    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        normalized_image = reinhard(image, target_mean, target_std)
        
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, cv2.cvtColor(normalized_image, cv2.COLOR_RGB2BGR))

normalize_images(INPUT_DIR, REFERENCE_GRID)
