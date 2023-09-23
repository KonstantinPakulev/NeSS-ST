import os
import argparse

from SensorData import SensorData


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path')

    args = parser.parse_args()

    dataset_path = args.dataset_path

    for subdir in ['scans', 'scans_test']:
        subdir_path = os.path.join(dataset_path, subdir)

        dirs = os.listdir(subdir_path)

        for s in dirs:
            scan_path = os.path.join(subdir_path, s, f"{s}.sens")

            sd = SensorData(scan_path)
            sd.export_depth_images(os.path.join(subdir_path, s, 'depth'))
            sd.export_color_images(os.path.join(subdir_path, s, 'color'))
            sd.export_poses(os.path.join(subdir_path, s, 'pose'))
            sd.export_intrinsics(os.path.join(subdir_path, s, 'intrinsic'))


if __name__ == '__main__':
    main()
