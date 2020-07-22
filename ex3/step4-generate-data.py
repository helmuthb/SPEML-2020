#! /usr/bin/env python3

"""Generate synthetic datasets

The fourth step is using the trained GAN for creating synthetic datasets.
"""

import argparse
import configparser
import glob
import os
import subprocess
import sys


def main():
    """Main procedure for creation of synthetic datasets.

    It will run through all classes and create matching
    synthetic data sets.
    The following command line arguments are supported:
    --generator: script to call
    --gans: path to the GANs folder
    --synthetic: path to the results folder
    --size: image size (part of folder name)
    --samplecount: number of samples to create
    """

    # read config file with defaults
    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation()
    )
    config.read('config.ini')
    # first step: parse config-file related arguments
    cfg_parser = argparse.ArgumentParser(
        description='Generate Synthetic Datasets',
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
    parser.add_argument('--generator', type=str,
                        help='script to call')
    parser.add_argument('--gans', type=str,
                        help='path to the GANs')
    parser.add_argument('--synthetic', type=str,
                        help='path to the results folder')
    parser.add_argument('--size', type=str,
                        help='image size (part of folder name)')
    parser.add_argument('--samplecount', type=int,
                        help='number of samples to generate')
    # parse command line arguments
    args = parser.parse_args()
    # does the source folder exist?
    gan_path = f"{args.gans}_{args.size}/"
    if not os.path.isdir(gan_path):
        print(f"Error: source folder `{gan_path}` does not exist!",
              file=sys.stderr)
        os._exit(1)
    # run through classes
    files = ["" for i in range(5)]
    for cls in range(5):
        # source path for this class
        source_pattern = f"{gan_path}/*-stylegan2-{cls}-*/network-final.pkl"
        source_files = glob.glob(source_pattern)
        if len(source_files) != 1:
            print(f"Error: pattern `{source_pattern}` has "
                  f"{len(source_files)} results!",
                  file=sys.stderr)
            os._exit(2)
        files[cls] = source_files[0]
    # second round - run through classes
    for cls in range(5):
        # call script
        print(f"Creating {args.samplecount} samples for class {cls}...")
        target_folder = f"{args.synthetic}_{args.size}/{cls}"
        subprocess.run([sys.executable, args.generator,
                        "generate-images",
                        f"--seeds=1-{args.samplecount}",
                        f"--network={files[cls]}",
                        f"--result-dir={target_folder}"])


# call main function if called
if __name__ == "__main__":
    main()
