#! /usr/bin/env python3

"""Create multi-resolution TFRecords from images.

As a second step, for each class the set of images has
to be converted into multi-resolution TFRecords,
such that StyleGAN2 can be trained against it.
"""

import argparse
import configparser
import os
import subprocess
import sys


def main():
    """Main procedure for creation of TFRecords.

    It will run through all classes and create matching
    folders for each class, with multi-resolution TFRecords.
    The following command line arguments are supported:
    --datatool: script to call
    --preprocessed: path to the pre-processed files
    --size: image size (part of folder name)
    --tfrecords: path to the target folder
    Defaults can be provided in `config.ini`.
    """

    # read config file with defaults
    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation()
    )
    config.read('config.ini')
    # first step: parse config-file related arguments
    cfg_parser = argparse.ArgumentParser(
        description='Create TFRecords',
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
    parser.add_argument('--datatool', type=str,
                        help='script to call')
    parser.add_argument('--preprocessed', type=str,
                        help='path to the pre-processed images')
    parser.add_argument('--size', type=str,
                        help='image size (part of folder name)')
    parser.add_argument('--tfrecords', type=str,
                        help='path to the target folder')
    # parse command line arguments
    args = parser.parse_args()
    # does the source folder exist?
    train_path = f"{args.preprocessed}_{args.size}/train"
    if not os.path.isdir(train_path):
        print(f"Error: source folder `{train_path}` does not exist!",
              file=sys.stderr)
        os._exit(1)
    # run through classes
    for cls in range(5):
        # source path for this class
        source_path = train_path + "/" + str(cls)
        if not os.path.isdir(source_path):
            print(f"Error: Class folder `{source_path}` not found",
                  file=sys.stderr)
            os._exit(2)
        # target path for this class
        target_path = f"{args.tfrecords}_{args.size}/" + str(cls)
        # create folder if not existing
        os.makedirs(target_path, exist_ok=True)
        # start existing script
        print(f"Creating TFRecords for class {cls}...")
        subprocess.run([sys.executable, args.datatool, "create_from_images",
                        target_path, source_path])


# call main function if called
if __name__ == "__main__":
    main()
