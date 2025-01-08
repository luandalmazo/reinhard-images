DIR = './Test_4cl_amostra/'
MATRIX_IMAGES_DIR = './matrix_images/'

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import shutil

os.makedirs(MATRIX_IMAGES_DIR, exist_ok=True)

def generate_matrix():
    dirs = os.listdir(DIR)
    matrix = np.zeros((4, 4), dtype=object)

    for i in range(4):
        dir_path = os.path.join(DIR, dirs[i])
        if os.path.isdir(dir_path):
            images = [f for f in os.listdir(dir_path) if f.endswith('.png')]
            images = images[:4]

            for j in range(4):
                image_path = os.path.join(dir_path, images[j])
                matrix[i][j] = Image.open(image_path)
                shutil.move(image_path, os.path.join(MATRIX_IMAGES_DIR, os.path.basename(image_path)))
                print(f"Moved: {image_path} -> {os.path.join(MATRIX_IMAGES_DIR, os.path.basename(image_path))}")

    return matrix

def save_matrix_as_figure(matrix, output_path):
    
    resized_images = [[img.resize((250, 250)) for img in row] for row in matrix]

    fig, ax = plt.subplots(4, 4, figsize=(8, 8))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)  # Remove all spaces

    for i in range(4):
        for j in range(4):
            img = resized_images[i][j]
            ax[i, j].imshow(img)
            ax[i, j].axis('off')
    
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

matrix = generate_matrix()
output_path = './matrix.png'
save_matrix_as_figure(matrix, output_path)
