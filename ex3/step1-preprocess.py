#! /usr/bin/env python3

"""Preprocessing of images for GAN and classifier.

As a first step, the following preparational steps
have to be performed on the dataset:
    * Image resize
      All images have to be resized to squares, with
      length being a power of two.
      For the development the resolution 128x128 has been
      chosen, for the real test 1024x1024.
    * Split into the classes
      Using the ground-truth labels, the images
      are split into the five classes.
    * Split into training and test dataset
      The data set is split into a training and a
      test data set. To ensure a good representation
      of each class we are using _stratified_ sampling.
"""

import os
import random
import argparse
import configparser
import sys
import time

from PIL import Image


def progressbar(it, prefix="", size=40, file=sys.stdout):
    """Display progress bar when working through list.

    This function is inspired by https://stackoverflow.com/a/34482761
    and displays a simple progress bar while running through a list.
    """

    count = len(it)
    last = time.time()

    def show(i, force=False):
        nonlocal last
        if force or time.time() > last+0.1:
            last = time.time()
            show_ratio(i / count, i)

    def show_ratio(r, i):
        x = int(size*r)
        percent = 100.*r
        done_str = "#" * x
        pending_str = "." * (size - x)
        file.write(f"{prefix}: [{done_str}{pending_str}] {percent:5.1f}% - "
                   f"{i}/{count}\r")
        file.flush()

    show(0, True)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    show(count, True)
    file.write("\n")
    file.flush()


def resize_square(src: str, trg: str, size: int) -> None:
    """Resize an image to a square with given size.

    The source image will be read, and converted into a square image
    with the length given in size.
    The result is saved in the file with the provided target name.
    The code has been inspired by python-resize-image.

    Args:
        src (str): The file location of the source image
        trg (str): The file location of the target image
        size (int): The target file size (a square of size x size)
    """

    # read image
    img = Image.open(src)
    # get minimum size
    min_size = min(img.size)
    # get bounding box of square
    left = (img.size[0]-min_size)//2
    top = (img.size[1]-min_size)//2
    square_box = (left, top, left+min_size, top+min_size)
    # resize using LANCZOS
    resized = img.resize(size=(size, size),
                         resample=Image.LANCZOS,
                         box=square_box)
    # save file
    resized.save(trg)
    # close all images
    img.close()
    resized.close()


def sort_resize(src: str, num: int, base: str, size: int,
                cls: int, in_train: bool) -> None:
    """Resize an image and store according to class and test/train.

    This functions takes an image, resizes it to the target size,
    and according to the class it belongs as well as whether it is
    a test or a training image the resized image will be stored
    into a folder like it will be copied into
    to it will be sorted into a folder structure like
    `/basefolder_<size>/<train|test>/<class>`
    with a file name like `img_<num>.png`.

    Args:
        src (str): The path of the source image
        num (int): Running number for the image name
        base (str): Base folder for the resized images
        size (int): Target size
        cls (int): Class of the image
        in_train (bool): Whether the image is in the training set
    """

    # construct target folder name
    t_folder = "train" if in_train else "test"
    trg_folder = f"{base}_{size}/{t_folder}/{cls}"
    # create folder if not here yet
    os.makedirs(trg_folder, exist_ok=True)
    # construct target file name
    trg = f"{trg_folder}/img_{num}.png"
    # resize image and store to target location
    resize_square(src, trg, size)


def split_resize(src_folder: str, trg_base: str, img_size: int,
                 train_ratio: float, label_file: str) -> None:
    """Split labeled data set into training and test dataset and resize.

    This function will read in the label file, identify the class
    for each image, and assign it to training class according to
    the ratio.

    Args:
        src_folder (str): Name of the source folder where the images are
        trg_base (str): Name of the target base
        img_size (int): Size of the images after resize
        train_ratio (float): Ratio of training images
        label_file (str): Name of the file with the class labels
    """

    # for each class the list of samples
    class_samples = [[] for i in range(5)]
    # read in labels file
    with open(label_file) as in_file:
        for line in in_file:
            line = line.strip()
            name, cls = tuple(line.split(","))
            if cls.isnumeric():
                class_samples[int(cls)].append(name)
    # set seed
    random.seed(42)
    # sample training sets and resize
    num = 0
    for cls in range(5):
        samples = class_samples[cls]
        n = len(samples)
        n_train = int(round(train_ratio * n))
        # get training samples as a set
        train = set(random.sample(samples, n_train))
        # run through resize for each sample
        num_proc = 20
        proc_list = []
        for sample in progressbar(samples, f"Class {cls}"):
            src = f"{src_folder}/{sample}.jpeg"
            child_pid = os.fork()
            if child_pid:
                proc_list.append(child_pid)
            else:
                sort_resize(src, num, trg_base, img_size, cls, sample in train)
                os._exit(0)
            while len(proc_list) >= num_proc:
                pid, exit = os.wait()
                proc_list.remove(pid)
            num += 1
        while proc_list:
            pid, exit = os.wait()
            proc_list.remove(pid)


def main():
    """Main procedure for pre-processing.

    It supports the following command line arguments:
    -e, --env: base environment to use (in file `config.ini`)
    --images: path to the images folder
    --preprocessed: base for path to the folder of preprocessed images
    --labels: path to the labels file
    --split: ratio between training and test
    --size: images size to be used
    Defaults can be provided in `config.ini`.
    """

    # read config file with defaults
    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation()
    )
    config.read('config.ini')
    # first step: parse config-file related arguments
    cfg_parser = argparse.ArgumentParser(
        description='Preprocess images',
        add_help=False)
    cfg_parser.add_argument(
        '-e', '--env',
        type=str,
        default=configparser.DEFAULTSECT,
        help='base environment to use (in file `config.ini`)')
    # parse config file related args
    cfg_args = cfg_parser.parse_known_args()[0]
    # add defaults from environment
    defaults = dict(config.items(cfg_args.env))
    # add other argument parser arguments
    parser = argparse.ArgumentParser(parents=[cfg_parser])
    parser.set_defaults(**defaults)
    parser.add_argument(
        '--images',
        type=str,
        help='path to the images folder')
    parser.add_argument(
        '--preprocessed',
        type=str,
        help='base for path to the folder of preprocessed images')
    parser.add_argument(
        '--labels',
        type=str,
        help='path to the labels file')
    parser.add_argument(
        '--split',
        type=float,
        help='ratio between training and test')
    parser.add_argument(
        '--size',
        type=int,
        help='image size for resized images')
    # parse command line arguments
    args = parser.parse_args()
    # output options
    print(args)
    exit(1)
    # run conversion
    split_resize(args.source, args.target, args.size,
                 args.split, args.labels)


# call main function if called
if __name__ == "__main__":
    main()
