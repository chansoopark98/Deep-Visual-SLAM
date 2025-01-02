import os
import glob
import json
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

class DimlConverterTFRecordSingleProcess:
    def __init__(self, root_dir, save_dir, target_shape=(480, 720), max_workers=4):
        """
        Args:
            root_dir: 최상위 데이터 디렉토리. 내부에 'train' 폴더와 'valid' 폴더가 존재한다고 가정.
                      예: root_dir/train/*.zip, root_dir/valid/*.zip
            save_dir: TFRecord 결과가 저장될 상위 디렉토리.
            target_shape: (height, width) 리사이즈 목표 크기
            max_workers: 압축 해제에 사용할 스레드 수
        """
        self.root_dir = root_dir
        self.save_dir = os.path.join(save_dir, 'diml_tfrecord')
        os.makedirs(self.save_dir, exist_ok=True)

        self.target_shape = target_shape
        self.max_workers = max_workers

        # 압축 풀린 원본 이미지 저장 경로
        self.raw_train_dir = os.path.join(self.root_dir, 'train_raw')
        self.raw_valid_dir = os.path.join(self.root_dir, 'valid_raw')
        os.makedirs(self.raw_train_dir, exist_ok=True)
        os.makedirs(self.raw_valid_dir, exist_ok=True)

        # TFRecord 최종 저장 경로
        self.train_tfrecord_path = os.path.join(self.save_dir, 'train.tfrecord')
        self.valid_tfrecord_path = os.path.join(self.save_dir, 'valid.tfrecord')

        # 1) TRAIN 세트 처리
        self._unzip_all(
            zip_src_dir=os.path.join(self.root_dir, 'train'),
            unzip_save_dir=self.raw_train_dir,
            split_name='train'
        )
        self._remove_unnecessary_folders(self.raw_train_dir, ['raw_png', 'warp_png'])
        train_count = self._convert_to_tfrecord(
            raw_file_dir=self.raw_train_dir,
            tfrecord_path=self.train_tfrecord_path
        )

        # 2) VALID 세트 처리
        self._unzip_all(
            zip_src_dir=os.path.join(self.root_dir, 'valid'),
            unzip_save_dir=self.raw_valid_dir,
            split_name='valid'
        )
        self._remove_unnecessary_folders(self.raw_valid_dir, ['raw_png', 'warp_png'])
        valid_count = self._convert_to_tfrecord(
            raw_file_dir=self.raw_valid_dir,
            tfrecord_path=self.valid_tfrecord_path
        )

        # 3) 메타데이터 저장
        metadata = {
            "train_count": train_count,
            "valid_count": valid_count
        }
        with open(os.path.join(self.save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

    # ------------------------------------------------------------------------
    # 1. ZIP 해제 (병렬)
    # ------------------------------------------------------------------------
    def _unzip_all(self, zip_src_dir, unzip_save_dir, split_name='train'):
        """
        zip_src_dir 안에 있는 모든 .zip 파일을 찾아 unzip_save_dir로 병렬 해제
        split_name: 'train' 혹은 'valid' 등 진행 상황 표시용
        """
        zip_files = glob.glob(os.path.join(zip_src_dir, '*.zip'))
        if not zip_files:
            print(f"[{split_name.upper()}] No zip files found in {zip_src_dir}")
            return

        def unzip_one_file(zip_path):
            zip_file_name = os.path.splitext(os.path.basename(zip_path))[0]
            target_dir = os.path.join(unzip_save_dir, zip_file_name)
            if os.path.exists(target_dir):
                return f"[{split_name}] Skip (already exists): {zip_path}"

            os.makedirs(target_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(target_dir)
            return f"[{split_name}] Unzipped: {zip_path}"

        results = []
        # 병렬 스레드풀
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {
                executor.submit(unzip_one_file, z): z for z in zip_files
            }
            for future in tqdm(
                as_completed(future_map),
                total=len(future_map),
                desc=f"[{split_name.upper()}] Unzipping files"
            ):
                results.append(future.result())

        for r in results:
            print(r)

    # ------------------------------------------------------------------------
    # 2. 불필요 폴더 제거 (재귀적 탐색)
    # ------------------------------------------------------------------------
    def _remove_unnecessary_folders(self, base_dir, folder_names):
        """base_dir 아래 folder_names 목록에 해당하는 폴더를 재귀적으로 찾아 삭제."""
        for folder_name in folder_names:
            targets = glob.glob(os.path.join(base_dir, '**', folder_name), recursive=True)
            for t in targets:
                print(f"Removing {t}")
                shutil.rmtree(t, ignore_errors=True)

    # ------------------------------------------------------------------------
    # 3. TFRecord 변환 (단일 프로세스)
    # ------------------------------------------------------------------------
    def _convert_to_tfrecord(self, raw_file_dir, tfrecord_path):
        """
        raw_file_dir 내부를 순회하며 (rgb, depth) 쌍을 찾고
        단일 프로세스로 TFRecord 형태로 저장.
        """
        raw_files = glob.glob(os.path.join(raw_file_dir, '*'))  # 1차 디렉토리
        file_pairs = []

        for raw_file in tqdm(raw_files, desc=f"[Collect] {tfrecord_path}"):
            raw_folds = glob.glob(os.path.join(raw_file, '*'))  # 2차 디렉토리

            for raw_fold in raw_folds:
                raw_scenes = glob.glob(os.path.join(raw_fold, '*'))  # 3차 디렉토리

                for raw_scene in raw_scenes:
                    rgb_path = os.path.join(raw_scene, 'col')
                    if not os.path.exists(rgb_path):
                        continue

                    rgb_files = sorted(glob.glob(os.path.join(rgb_path, '*.png')))

                    used_files = set()
                    # 3프레임 간격 샘플링
                    for idx in range(0, len(rgb_files), 3):
                        rgb_name = rgb_files[idx]
                        depth_name = rgb_name.replace('col', 'up_png').replace('c.png', 'ud.png')

                        if not os.path.exists(depth_name):
                            continue

                        used_files.add(rgb_name)
                        used_files.add(depth_name)
                        file_pairs.append((rgb_name, depth_name))

                    # 사용하지 않는 파일 삭제
                    for f in rgb_files:
                        if f not in used_files:
                            depth_file = f.replace('col', 'up_png').replace('c.png', 'ud.png')
                            if os.path.exists(f):
                                os.remove(f)
                            if os.path.exists(depth_file):
                                os.remove(depth_file)

        # 실제 TFRecord 생성: 단일 프로세스로 순차 처리
        self._write_tfrecord(file_pairs, tfrecord_path)
        return len(file_pairs)

    def _write_tfrecord(self, file_pairs, tfrecord_path):
        """
        (rgb_name, depth_name) 쌍을 순차적으로 처리해 TFRecord에 기록.
        병렬처리 X, 단일 프로세스.
        """
        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            for (rgb_name, depth_name) in tqdm(file_pairs, desc=f"[Write] {os.path.basename(tfrecord_path)}"):
                rgb_bytes, depth_bytes = self._process_image_pair((rgb_name, depth_name))
                example = self._serialize_example(rgb_bytes, depth_bytes)
                writer.write(example)

    # ------------------------------------------------------------------------
    # (rgb, depth) pair 처리 (단일 프로세스)
    # ------------------------------------------------------------------------
    def _process_image_pair(self, file_pair):
        """
        [단일 프로세스용]
        (rgb_file, depth_file)을 로드 & 리사이즈 & 직렬화용 bytes로 변환.
        """
        rgb_file, depth_file = file_pair

        # 1) 이미지 로드
        rgb = np.array(Image.open(rgb_file))
        depth = np.array(Image.open(depth_file)) * 0.001  # mm -> m 스케일링 예시

        # 2) 텐서 변환 & 리사이즈
        rgb_tensor = tf.convert_to_tensor(rgb, tf.uint8)
        depth_tensor = tf.convert_to_tensor(depth, tf.float32)
        depth_tensor = tf.expand_dims(depth_tensor, axis=-1)

        rgb_tensor = tf.image.resize(rgb_tensor, self.target_shape, method='bilinear')
        depth_tensor = tf.image.resize(depth_tensor, self.target_shape, method='nearest')

        # 3) 타입 캐스팅
        depth_tensor = tf.squeeze(depth_tensor, axis=-1)
        rgb_tensor = tf.cast(rgb_tensor, tf.uint8)
        depth_tensor = tf.cast(depth_tensor, tf.float16)

        # 4) 바이트 인코딩
        rgb_bytes = tf.io.encode_jpeg(rgb_tensor, quality=100).numpy()
        depth_bytes = tf.io.serialize_tensor(depth_tensor).numpy()

        return rgb_bytes, depth_bytes

    # ------------------------------------------------------------------------
    # 최종 TFRecord 직렬화 예시
    # ------------------------------------------------------------------------
    def _serialize_example(self, rgb_bytes, depth_bytes):
        feature = {
            'rgb': tf.train.Feature(bytes_list=tf.train.BytesList(value=[rgb_bytes])),
            'depth': tf.train.Feature(bytes_list=tf.train.BytesList(value=[depth_bytes]))
        }
        return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


# ----------------------------------------------------------------------------
#  메인 실행
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    # 디렉토리 구조
    # root_dir/
    #   ├─ train/
    #   │   ├─ diml_trainA.zip
    #   │   ├─ diml_trainB.zip
    #   │   └─ ...
    #   └─ valid/
    #       ├─ diml_validA.zip
    #       ├─ diml_validB.zip
    #       └─ ...
    #
    # 위 구조에서 압축을 풀어
    # root_dir/train_raw/
    # root_dir/valid_raw/
    # ... 구조로 배치 후,
    # 최종적으로 save_dir/diml_tfrecord/train.tfrecord, valid.tfrecord 생성

    root_dir = '/media/park-ubuntu/park_file/depth_data/'
    save_dir = './depth/data/new/'

    converter = DimlConverterTFRecordSingleProcess(
        root_dir=root_dir,
        save_dir=save_dir,
        target_shape=(480, 720),
        max_workers=4  # 압축 해제 시 스레드 개수
    )
