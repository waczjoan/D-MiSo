import os
import shutil
import re
from argparse import ArgumentParser

def restructure_images_folder(img_folder):
    pattern = re.compile(r'^cam(\d{2})_(\d{4})\.png$')

    for filename in os.listdir(img_folder):
        match = pattern.match(filename)
        if match:
            cam_id = match.group(1)
            cam_folder = os.path.join(img_folder, f'cam{cam_id}')
            os.makedirs(cam_folder, exist_ok=True)

            src_path = os.path.join(img_folder, filename)
            dst_path = os.path.join(cam_folder, filename)
            shutil.move(src_path, dst_path)
            print(f"Moved {filename} to {cam_folder}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Restructure N3D images folder")
    parser.add_argument('--img_folder', required=True, type=str, help='Source folder containing images')
    args = parser.parse_args()

    restructure_images_folder(args.img_folder)
