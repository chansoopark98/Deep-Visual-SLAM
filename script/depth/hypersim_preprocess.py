import argparse
import os
import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from pylab import count_nonzero, clip

IMG_WIDTH = 1024
IMG_HEIGHT = 768
FOCAL_LENGTH = 886.81

def tone_map(rgb, entity_id_map):
    """
    Adapted from:
    https://github.com/apple/ml-hypersim/blob/main/code/python/tools/scene_generate_images_tonemap.py
    """
    
    gamma = 1.0 / 2.2  # standard gamma correction exponent
    inv_gamma = 1.0 / gamma
    percentile = 90
    brightness_nth_percentile_desired = 0.8

    valid_mask = entity_id_map != -1
    if count_nonzero(valid_mask) == 0:
        scale = 1.0
    else:
        brightness = (
            0.3 * rgb[:, :, 0] + 0.59 * rgb[:, :, 1] + 0.11 * rgb[:, :, 2]
        )
        brightness_valid = brightness[valid_mask]
        eps = 0.0001
        brightness_nth_percentile_current = np.percentile(brightness_valid, percentile)

        if brightness_nth_percentile_current < eps:
            scale = 0.0
        else:
            scale = (
                np.power(brightness_nth_percentile_desired, inv_gamma)
                / brightness_nth_percentile_current
            )

    rgb_color_tm = np.power(np.maximum(scale * rgb, 0), gamma)
    rgb_color_tm = clip(rgb_color_tm, 0, 1)
    return rgb_color_tm


def dist_2_depth(width, height, flt_focal, distance):
    """
    According to:
    https://github.com/apple/ml-hypersim/issues/9
    """
    img_plane_x = (
        np.linspace((-0.5 * width) + 0.5, (0.5 * width) - 0.5, width)
        .reshape(1, width)
        .repeat(height, 0)
        .astype(np.float32)[:, :, None]
    )
    img_plane_y = (
        np.linspace((-0.5 * height) + 0.5, (0.5 * height) - 0.5, height)
        .reshape(height, 1)
        .repeat(width, 1)
        .astype(np.float32)[:, :, None]
    )
    img_plane_z = np.full([height, width, 1], flt_focal, np.float32)
    img_plane = np.concatenate([img_plane_x, img_plane_y, img_plane_z], 2)

    depth = distance / np.linalg.norm(img_plane, 2, 2) * flt_focal
    return depth


def process_row(
    i,
    row,
    dataset_dir,
    split_output_dir,
    img_width,
    img_height,
    focal_length,
):
    """
    한 개의 row(= 장면 + 카메라 + 프레임)에 대한 전처리를 수행하고
    해당 row에 기록할 메타데이터를 반환합니다.
    """
    # 출력용 dict (DataFrame에 다시 할당)
    row_result = {
        "index": i,
        "rgb_path": None,
        "rgb_mean": None,
        "rgb_std": None,
        "rgb_min": None,
        "rgb_max": None,
        "depth_path": None,
        "depth_mean": None,
        "depth_std": None,
        "depth_min": None,
        "depth_max": None,
        "invalid_ratio": None,
    }
    
    # 파일 경로 구성
    rgb_path_in = os.path.join(
        dataset_dir,
        row.scene_name,
        "images",
        f"scene_{row.camera_name}_final_hdf5",
        f"frame.{row.frame_id:04d}.color.hdf5",
    )
    dist_path_in = os.path.join(
        dataset_dir,
        row.scene_name,
        "images",
        f"scene_{row.camera_name}_geometry_hdf5",
        f"frame.{row.frame_id:04d}.depth_meters.hdf5",
    )
    render_entity_id_path_in = os.path.join(
        dataset_dir,
        row.scene_name,
        "images",
        f"scene_{row.camera_name}_geometry_hdf5",
        f"frame.{row.frame_id:04d}.render_entity_id.hdf5",
    )

    # 실제 파일 존재 확인
    assert os.path.exists(rgb_path_in), f"Not found: {rgb_path_in}"
    assert os.path.exists(dist_path_in), f"Not found: {dist_path_in}"
    assert os.path.exists(render_entity_id_path_in), f"Not found: {render_entity_id_path_in}"

    # HDF5 로드
    with h5py.File(rgb_path_in, "r") as f:
        rgb = np.array(f["dataset"]).astype(float)
    with h5py.File(dist_path_in, "r") as f:
        dist_from_center = np.array(f["dataset"]).astype(float)
    with h5py.File(render_entity_id_path_in, "r") as f:
        render_entity_id = np.array(f["dataset"]).astype(int)

    # Tone map
    rgb_color_tm = tone_map(rgb, render_entity_id)
    rgb_int = (rgb_color_tm * 255).astype(np.uint8)  # [H, W, RGB]

    # Distance -> depth
    plane_depth = dist_2_depth(img_width, img_height, focal_length, dist_from_center)
    valid_mask = render_entity_id != -1
    invalid_ratio = (np.prod(valid_mask.shape) - valid_mask.sum()) / np.prod(valid_mask.shape)
    plane_depth[~valid_mask] = 0

    # 저장 경로 생성
    scene_path_out = os.path.join(split_output_dir, row.scene_name)
    os.makedirs(scene_path_out, exist_ok=True)

    # RGB 저장
    rgb_name = f"rgb_{row.camera_name}_fr{row.frame_id:04d}.png"
    rgb_path_out = os.path.join(scene_path_out, rgb_name)
    cv2.imwrite(rgb_path_out, cv2.cvtColor(rgb_int, cv2.COLOR_RGB2BGR))

    # Depth 저장 (mm 단위, uint16 변환)
    plane_depth_mm = plane_depth * 1000.0
    plane_depth_mm = plane_depth_mm.astype(np.uint16)
    depth_name = f"depth_plane_{row.camera_name}_fr{row.frame_id:04d}.png"
    depth_path_out = os.path.join(scene_path_out, depth_name)
    cv2.imwrite(depth_path_out, plane_depth_mm)

    # 메타데이터 계산
    row_result["rgb_path"] = os.path.join(row.scene_name, rgb_name)  # 상대경로 기록
    row_result["rgb_mean"] = float(np.mean(rgb_int))
    row_result["rgb_std"] = float(np.std(rgb_int))
    row_result["rgb_min"] = float(np.min(rgb_int))
    row_result["rgb_max"] = float(np.max(rgb_int))

    restored_depth = plane_depth_mm / 1000.0
    row_result["depth_path"] = os.path.join(row.scene_name, depth_name)  # 상대경로 기록
    row_result["depth_mean"] = float(np.mean(restored_depth))
    row_result["depth_std"] = float(np.std(restored_depth))
    row_result["depth_min"] = float(np.min(restored_depth))
    row_result["depth_max"] = float(np.max(restored_depth))

    row_result["invalid_ratio"] = float(invalid_ratio)

    return row_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_csv",
        type=str,
        default="./depth/util/metadata_images_split_scene_v1.csv",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/media/park-ubuntu/park_file/hypersim/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/media/park-ubuntu/park_file/hypersim_output/",
    )
    args = parser.parse_args()

    split_csv = args.split_csv
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir

    raw_meta_df = pd.read_csv(split_csv)
    meta_df = raw_meta_df[raw_meta_df.included_in_public_release].copy()

    for split in ["train", "val", "test"]:
        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)

        split_meta_df = meta_df[meta_df.split_partition_name == split].copy()

        # 준비: 병렬 실행 결과를 저장할 컬럼을 미리 만들거나, 
        #       나중에 dictionary -> DataFrame으로 merge할 수 있습니다.
        split_meta_df["rgb_path"] = None
        split_meta_df["rgb_mean"] = np.nan
        split_meta_df["rgb_std"] = np.nan
        split_meta_df["rgb_min"] = np.nan
        split_meta_df["rgb_max"] = np.nan
        split_meta_df["depth_path"] = None
        split_meta_df["depth_mean"] = np.nan
        split_meta_df["depth_std"] = np.nan
        split_meta_df["depth_min"] = np.nan
        split_meta_df["depth_max"] = np.nan
        split_meta_df["invalid_ratio"] = np.nan

        # 병렬 처리
        # n_jobs = -1은 CPU 모든 코어 사용
        results = Parallel(n_jobs=-1)(
            delayed(process_row)(
                i,
                row,
                dataset_dir,
                split_output_dir,
                IMG_WIDTH,
                IMG_HEIGHT,
                FOCAL_LENGTH,
            )
            for i, row in tqdm(split_meta_df.iterrows(), total=len(split_meta_df))
        )

        # 결과를 split_meta_df에 반영
        for res in results:
            i = res["index"]
            split_meta_df.at[i, "rgb_path"] = res["rgb_path"]
            split_meta_df.at[i, "rgb_mean"] = res["rgb_mean"]
            split_meta_df.at[i, "rgb_std"] = res["rgb_std"]
            split_meta_df.at[i, "rgb_min"] = res["rgb_min"]
            split_meta_df.at[i, "rgb_max"] = res["rgb_max"]
            split_meta_df.at[i, "depth_path"] = res["depth_path"]
            split_meta_df.at[i, "depth_mean"] = res["depth_mean"]
            split_meta_df.at[i, "depth_std"] = res["depth_std"]
            split_meta_df.at[i, "depth_min"] = res["depth_min"]
            split_meta_df.at[i, "depth_max"] = res["depth_max"]
            split_meta_df.at[i, "invalid_ratio"] = res["invalid_ratio"]

        # filename_list_{split}.txt 저장
        with open(os.path.join(split_output_dir, f"filename_list_{split}.txt"), "w+") as f:
            lines = split_meta_df.apply(
                lambda r: f"{r['rgb_path']} {r['depth_path']}", axis=1
            ).tolist()
            f.write("\n".join(lines))

        # filename_meta_{split}.csv 저장
        split_meta_df.to_csv(
            os.path.join(split_output_dir, f"filename_meta_{split}.csv"), 
            header=True, 
            index=False
        )

    print("Preprocess finished")