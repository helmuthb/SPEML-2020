#! /usr/bin/env python3

"""Train network for GAN

The third step is training the GAN with the provided data set.
"""

import argparse
import configparser
import os
import subprocess
import sys


def main():
    """Main procedure for training the GAN.

    It will run through all classes and train the GANs
    for each class.
    The following command line arguments are supported:
    --traintool: script to call
    --tfrecords: path to the TFrecords
    --gans: path to the GANs folder
    --size: image size (part of folder name)
    --gpus: number of GPUs to use
    --kimg: number of training rounds (in 1000s of images)
    """

    # read config file with defaults
    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation()
    )
    config.read('config.ini')
    # first step: parse config-file related arguments
    cfg_parser = argparse.ArgumentParser(
        description='Train GAN',
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
    parser.add_argument('--traintool', type=str,
                        help='script to call')
    parser.add_argument('--tfrecords', type=str,
                        help='path to the TFrecords')
    parser.add_argument('--gans', type=str,
                        help='path to the GANs folder')
    parser.add_argument('--size', type=str,
                        help='image size (part of folder name)')
    parser.add_argument('--gpus', type=int,
                        help='number of GPUs to use')
    parser.add_argument('--kimg', type=int,
                        help='number of training rounds (in 1000s of images)')
    # parse command line arguments
    args = parser.parse_args()
    # does the source folder exist?
    train_path = f"{args.tfrecords}_{args.size}/"
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
        # start existing script
        print(f"Training GAN for class {cls}...")
        subprocess.run([sys.executable, args.traintool,
                        f"--num-gpus={args.gpus}",
                        f"--data-dir={train_path}",
                        f"--dataset={cls}",
                        f"--total-kimg={args.kimg}",
                        f"--result-dir={args.gans}_{args.size}",
                        "--config=config-f",
                        "--mirror-augment=true"])


# call main function if called
if __name__ == "__main__":
    main()
