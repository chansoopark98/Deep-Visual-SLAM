
import os
import argparse
from collections import defaultdict


def default_parser():
    parser = argparse.ArgumentParser("Depth estimate dataset installer")
    parser.add_argument('--root-path', type=str, default='depth/data',help='dist dataset root path')
    parser.add_argument('--nyu-v2', action='store_true', help='data preprocess Nyu_Depth_V2 dataset')
    parser.add_argument('--diode', action='store_true', help='data preprocess DIODE dataset')
    parser.add_argument('--diml', action='store_true', help='data preprocess diml dataset')
    parser.add_argument('--hyersim', action='store_true', help='data preprocess hyersim dataset')
    args = parser.parse_args()
    return args


def main(args):
    dist_path_dict = defaultdict()

    if args.nyu_v2:
        from script.depth.tfds_nyu_converter_tfrecord import nyu_main_process
        nyu_dist_path = nyu_main_process(args)
        dist_path_dict['nyu'] = nyu_dist_path

    if args.diode:
        from script.depth.diode_converter_tfrecord import DiodeConverterTFRecord
        DiodeConverterTFRecord(args.root_path)
        dist_path_dict['diode'] = os.path.join(args.root_path, "diode")
        del DiodeConverterTFRecord

    if args.diml:
        raise NotImplementedError

    if args.hyersim:
        raise NotImplementedError

    for k, v in dist_path_dict.items():
        print(f'{k} dataset install Path: {v}')


if __name__ == '__main__':
    args = default_parser()
    main(args)