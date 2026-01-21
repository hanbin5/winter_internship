#!/usr/bin/env python3
"""
UZH-FPV 데이터셋을 ICL-NUIM 형식으로 변환하는 스크립트

ICL-NUIM 형식:
- rgb/ 폴더에 0.png, 1.png, ... 형태로 이미지 저장
- groundtruth.txt: frame_id tx ty tz qx qy qz qw
- associations.txt: timestamp depth/N.png timestamp rgb/N.png

조건:
- pose가 NaN이 아닌 유효한 값을 가지는 이미지만 저장
- 이미지 개수 == pose 개수
"""

import os
import argparse
import cv2
import numpy as np
import math
from pathlib import Path
from rosbags.highlevel import AnyReader


def is_valid_pose(pose):
    """pose 값이 유효한지 (NaN이 아닌지) 확인"""
    values = [pose['x'], pose['y'], pose['z'],
              pose['qx'], pose['qy'], pose['qz'], pose['qw']]
    return all(not math.isnan(v) for v in values)


def find_closest_pose(img_timestamp, poses, max_time_diff=0.1):
    """
    이미지 타임스탬프에 가장 가까운 유효한 pose 찾기
    max_time_diff: 최대 허용 시간 차이 (초)
    """
    best_pose = None
    best_diff = float('inf')

    for pose in poses:
        if not is_valid_pose(pose):
            continue

        time_diff = abs(pose['timestamp'] - img_timestamp)
        if time_diff < best_diff:
            best_diff = time_diff
            best_pose = pose

    if best_pose is None or best_diff > max_time_diff:
        return None, best_diff

    return best_pose, best_diff


def convert_to_icl_format(bag_file, output_dir, image_topic, pose_topic, max_time_diff=0.1):
    """
    rosbag을 ICL-NUIM 형식으로 변환
    pose가 유효하지 않은 이미지는 저장하지 않음
    """
    rgb_dir = os.path.join(output_dir, 'rgb')
    os.makedirs(rgb_dir, exist_ok=True)

    images = []
    poses = []

    print(f"Reading bag file: {bag_file}")
    print(f"Image topic: {image_topic}")
    print(f"Pose topic: {pose_topic}")

    with AnyReader([Path(bag_file)]) as reader:
        for connection, timestamp, rawdata in reader.messages():
            msg = reader.deserialize(rawdata, connection.msgtype)

            if connection.topic == image_topic:
                img_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                images.append({
                    'timestamp': img_timestamp,
                    'height': msg.height,
                    'width': msg.width,
                    'encoding': msg.encoding,
                    'data': msg.data
                })

            elif connection.topic == pose_topic:
                pose_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                # PoseStamped vs Odometry 처리
                if hasattr(msg, 'pose') and hasattr(msg.pose, 'pose'):
                    # Odometry 메시지
                    p = msg.pose.pose
                else:
                    # PoseStamped 메시지
                    p = msg.pose

                poses.append({
                    'timestamp': pose_timestamp,
                    'x': p.position.x,
                    'y': p.position.y,
                    'z': p.position.z,
                    'qx': p.orientation.x,
                    'qy': p.orientation.y,
                    'qz': p.orientation.z,
                    'qw': p.orientation.w
                })

    print(f"\nFound {len(images)} images and {len(poses)} poses")

    # 유효한 pose 개수 확인
    valid_poses = [p for p in poses if is_valid_pose(p)]
    print(f"Valid poses (non-NaN): {len(valid_poses)}")

    if len(valid_poses) == 0:
        print("Error: No valid poses found!")
        return None

    # timestamp로 정렬
    images.sort(key=lambda x: x['timestamp'])
    poses.sort(key=lambda x: x['timestamp'])

    # 매칭된 데이터만 저장
    matched_data = []
    skipped_count = 0

    for img in images:
        pose, time_diff = find_closest_pose(img['timestamp'], poses, max_time_diff)

        if pose is None:
            skipped_count += 1
            continue

        # 이미지 디코딩
        if img['encoding'] == 'bgr8':
            cv_img = np.frombuffer(img['data'], dtype=np.uint8).reshape(
                img['height'], img['width'], 3)
        elif img['encoding'] == 'rgb8':
            cv_img = np.frombuffer(img['data'], dtype=np.uint8).reshape(
                img['height'], img['width'], 3)
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        elif img['encoding'] == 'mono8':
            cv_img = np.frombuffer(img['data'], dtype=np.uint8).reshape(
                img['height'], img['width'])
            # mono8을 3채널로 변환
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
        else:
            print(f"Unknown encoding: {img['encoding']}, skipping...")
            skipped_count += 1
            continue

        matched_data.append({
            'image': cv_img,
            'img_timestamp': img['timestamp'],
            'pose': pose,
            'time_diff': time_diff
        })

    print(f"\nMatched frames: {len(matched_data)}")
    print(f"Skipped frames (no valid pose): {skipped_count}")

    if len(matched_data) == 0:
        print("Error: No matched frames!")
        return None

    # 이미지 저장 및 groundtruth 생성
    groundtruth_lines = []
    association_lines = []

    for i, data in enumerate(matched_data):
        # 이미지 저장
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(rgb_dir, filename), data['image'])

        # groundtruth 라인 (ICL-NUIM 형식: frame_id tx ty tz qx qy qz qw)
        pose = data['pose']
        gt_line = f"{i} {pose['x']} {pose['y']} {pose['z']} {pose['qx']} {pose['qy']} {pose['qz']} {pose['qw']}"
        groundtruth_lines.append(gt_line)

        # associations 라인 (ICL-NUIM 형식: timestamp depth/N.png timestamp rgb/N.png)
        # depth가 없으므로 rgb만 기록
        assoc_line = f"{i} rgb/{i}.png"
        association_lines.append(assoc_line)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(matched_data)} frames")

    # groundtruth.txt 저장
    gt_file = os.path.join(output_dir, 'groundtruth.txt')
    with open(gt_file, 'w') as f:
        for line in groundtruth_lines:
            f.write(line + '\n')
    print(f"\nSaved groundtruth to {gt_file}")

    # associations.txt 저장
    assoc_file = os.path.join(output_dir, 'associations.txt')
    with open(assoc_file, 'w') as f:
        for line in association_lines:
            f.write(line + '\n')
    print(f"Saved associations to {assoc_file}")

    # 통계
    time_diffs = [d['time_diff'] for d in matched_data]
    print(f"\n=== Statistics ===")
    print(f"Total frames saved: {len(matched_data)}")
    print(f"Average time diff: {np.mean(time_diffs)*1000:.2f} ms")
    print(f"Max time diff: {np.max(time_diffs)*1000:.2f} ms")
    print(f"Min time diff: {np.min(time_diffs)*1000:.2f} ms")

    return matched_data


def process_all_bags(data_dir, output_base_dir, image_topic, pose_topic, max_time_diff=0.1):
    """
    data_dir 내의 모든 rosbag 파일을 처리
    """
    data_path = Path(data_dir)
    bag_files = list(data_path.glob('**/*.bag'))

    print(f"Found {len(bag_files)} bag files")

    results = {}

    for bag_file in bag_files:
        print(f"\n{'='*60}")
        print(f"Processing: {bag_file}")
        print('='*60)

        # 출력 디렉토리 결정 (bag 파일명 기반)
        bag_name = bag_file.stem
        output_dir = os.path.join(output_base_dir, bag_name)

        try:
            matched = convert_to_icl_format(
                str(bag_file),
                output_dir,
                image_topic,
                pose_topic,
                max_time_diff
            )
            if matched:
                results[bag_name] = len(matched)
        except Exception as e:
            print(f"Error processing {bag_file}: {e}")
            results[bag_name] = 0

    print(f"\n{'='*60}")
    print("Summary")
    print('='*60)
    for name, count in results.items():
        print(f"{name}: {count} frames")


def main():
    parser = argparse.ArgumentParser(
        description="Convert UZH-FPV dataset to ICL-NUIM format"
    )
    parser.add_argument("input", help="Input bag file or directory containing bag files")
    parser.add_argument("output", help="Output directory")
    parser.add_argument("--image-topic", default="/snappy_cam/stereo_l",
                        help="Image topic (default: /snappy_cam/stereo_l)")
    parser.add_argument("--pose-topic", default="/groundtruth/pose",
                        help="Pose topic (default: /groundtruth/pose)")
    parser.add_argument("--max-time-diff", type=float, default=0.1,
                        help="Maximum time difference for matching (seconds, default: 0.1)")
    parser.add_argument("--batch", action="store_true",
                        help="Process all bag files in directory")

    args = parser.parse_args()

    if args.batch or os.path.isdir(args.input):
        process_all_bags(args.input, args.output,
                        args.image_topic, args.pose_topic, args.max_time_diff)
    else:
        convert_to_icl_format(args.input, args.output,
                             args.image_topic, args.pose_topic, args.max_time_diff)


if __name__ == '__main__':
    main()
