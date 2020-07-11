#! /usr/bin/env python3

"""Train network for GAN

The third step is training the GAN with the provided data set.
"""

import argparse
import os
import subprocess
import sys


def main():
    """Main procedure for creation of TFRecords.

    It will run through all classes and create matching
    folders for each class, with multi-resolution TFRecords.
    The following command line arguments are supported:
    --traintool: script to call
    --tfrecords: path to the TFrecords
    --results: path to the results folder
    --class: which class to train
    --gpus: number of GPUs to use
    --kimg: number of training rounds (in 1000s of images)
    """

    # create argument parser
    parser = argparse.ArgumentParser(description='Create TFRecords')
    parser.add_argument('--traintool', type=str,
                        help='script to call')
    parser.add_argument('--tfrecords', type=str,
                        help='path to the TFrecords')
    parser.add_argument('--results', type=str,
                        help='path to the results folder')
    parser.add_argument('--kimg', type=int,
                        help='number of training rounds (in 1000s of images)')
    parser.add_argument('--class', type=int, required=True,
                        help='which class to train')
    # parse command line arguments
    args = parser.parse_args()
    # does the source folder exist?
    train_path = args.source + "/train"
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
        target_path = args.target + "/" + str(cls)
        # create folder if not existing
        os.makedirs(target_path, exist_ok=True)
        # start existing script
        print(f"Creating TFRecords for class {cls}...")
        subprocess.run([sys.executable, args.script, "create_from_images",
                        target_path, source_path])


# call main function if called
if __name__ == "__main__":
    main()
