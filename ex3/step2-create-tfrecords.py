#! /usr/bin/env python3

"""Create multi-resolution TFRecords from images.

As a second step, for each class the set of images has
to be converted into multi-resolution TFRecords,
such that StyleGAN2 can be trained against it.
"""

import argparse
import os
import subprocess
import sys


def main():
    """Main procedure for creation of TFRecords.

    It will run through all classes and create matching
    folders for each class, with multi-resolution TFRecords.
    The following command line arguments are required:
    --script: script to call
    --source: path to the pre-processed files
    --target: path to the target folder
    """

    # create argument parser
    parser = argparse.ArgumentParser(description='Create TFRecords')
    parser.add_argument('--script', type=str,
                        default='./stylegan2/dataset_tool.py',
                        help='script to call')
    parser.add_argument('--source', type=str, required=True,
                        help='path to the pre-processed files')
    parser.add_argument('--target', type=str, required=True,
                        help='path to the target folder')
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
