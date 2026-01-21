import os
import argparse
import cv2
import numpy as np
from rosbags.rosbag1 import Reader
import struct

def deserialize_ros1_image(rawdata):
    """ROS1 Image 메시지를 직접 역직렬화"""
    offset = 0
    
    seq = struct.unpack_from('<I', rawdata, offset)[0]
    offset += 4
    stamp_sec = struct.unpack_from('<I', rawdata, offset)[0]
    offset += 4
    stamp_nsec = struct.unpack_from('<I', rawdata, offset)[0]
    offset += 4
    
    frame_id_len = struct.unpack_from('<I', rawdata, offset)[0]
    offset += 4
    frame_id = rawdata[offset:offset+frame_id_len].decode('utf-8')
    offset += frame_id_len
    
    height = struct.unpack_from('<I', rawdata, offset)[0]
    offset += 4
    width = struct.unpack_from('<I', rawdata, offset)[0]
    offset += 4
    
    encoding_len = struct.unpack_from('<I', rawdata, offset)[0]
    offset += 4
    encoding = rawdata[offset:offset+encoding_len].decode('utf-8')
    offset += encoding_len
    
    is_bigendian = struct.unpack_from('<B', rawdata, offset)[0]
    offset += 1
    step = struct.unpack_from('<I', rawdata, offset)[0]
    offset += 4
    
    data_len = struct.unpack_from('<I', rawdata, offset)[0]
    offset += 4
    data = rawdata[offset:offset+data_len]
    
    return {
        'timestamp': stamp_sec + stamp_nsec * 1e-9,
        'height': height,
        'width': width,
        'encoding': encoding,
        'data': data
    }

def deserialize_pose_or_odom(rawdata, msgtype):
    """Pose 또는 Odometry 메시지 파싱"""
    offset = 0
    
    # Header
    seq = struct.unpack_from('<I', rawdata, offset)[0]
    offset += 4
    stamp_sec = struct.unpack_from('<I', rawdata, offset)[0]
    offset += 4
    stamp_nsec = struct.unpack_from('<I', rawdata, offset)[0]
    offset += 4
    
    frame_id_len = struct.unpack_from('<I', rawdata, offset)[0]
    offset += 4
    offset += frame_id_len
    
    # Odometry일 경우 child_frame_id도 건너뛰기
    if 'odom' in msgtype:
        child_frame_len = struct.unpack_from('<I', rawdata, offset)[0]
        offset += 4
        offset += child_frame_len
    
    # Pose (position + orientation)
    x, y, z = struct.unpack_from('<ddd', rawdata, offset)
    offset += 24
    qx, qy, qz, qw = struct.unpack_from('<dddd', rawdata, offset)
    
    timestamp = stamp_sec + stamp_nsec * 1e-9
    
    return {
        'timestamp': timestamp,
        'x': x,
        'y': y,
        'z': z,
        'qx': qx,
        'qy': qy,
        'qz': qz,
        'qw': qw
    }

def extract_images_and_poses(bag_file, output_dir, image_topic, pose_topic):
    """이미지와 pose를 함께 추출"""
    os.makedirs(output_dir, exist_ok=True)
    
    images = []
    poses = []
    
    print("Reading bag file...")
    
    with Reader(bag_file) as reader:
        # 먼저 모든 데이터 수집
        for connection, timestamp, rawdata in reader.messages():
            try:
                if connection.topic == image_topic:
                    msg = deserialize_ros1_image(rawdata)
                    images.append(msg)
                    
                elif connection.topic == pose_topic:
                    pose = deserialize_pose_or_odom(rawdata, connection.msgtype)
                    poses.append(pose)
                    
            except Exception as e:
                continue
    
    print(f"Found {len(images)} images and {len(poses)} poses")
    
    # timestamp로 정렬
    images.sort(key=lambda x: x['timestamp'])
    poses.sort(key=lambda x: x['timestamp'])
    
    # 이미지 저장 및 매칭
    matched_data = []
    
    for i, img in enumerate(images):
        # 이미지 저장
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
        else:
            continue
        
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(output_dir, filename), cv_img)
        
        # 가장 가까운 pose 찾기
        img_time = img['timestamp']
        closest_idx = min(range(len(poses)), 
                         key=lambda j: abs(poses[j]['timestamp'] - img_time))
        
        pose = poses[closest_idx]
        time_diff = abs(pose['timestamp'] - img_time)
        
        matched_data.append({
            'frame_id': i,
            'timestamp': pose['timestamp'],
            'x': pose['x'],
            'y': pose['y'],
            'z': pose['z'],
            'qx': pose['qx'],
            'qy': pose['qy'],
            'qz': pose['qz'],
            'qw': pose['qw'],
            'time_diff': time_diff
        })
        
        if i % 100 == 0:
            print(f"Processed {i} images")
    
    # TUM 포맷으로 저장
    trajectory_file = os.path.join(output_dir, 'groundtruth.txt')
    with open(trajectory_file, 'w') as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for data in matched_data:
            f.write(f"{data['frame_id']} {data['x']:.6f} {data['y']:.6f} {data['z']:.6f} "
                   f"{data['qx']:.8f} {data['qy']:.8f} {data['qz']:.8f} {data['qw']:.6f}\n")
    
    print(f"\nSaved trajectory to {trajectory_file}")
    
    # 통계
    time_diffs = [d['time_diff'] for d in matched_data]
    print(f"\nMatching statistics:")
    print(f"  Total matched frames: {len(matched_data)}")
    print(f"  Average time difference: {np.mean(time_diffs)*1000:.2f} ms")
    print(f"  Max time difference: {np.max(time_diffs)*1000:.2f} ms")
    
    return matched_data

def main():
    parser = argparse.ArgumentParser(description="Extract images and trajectory from ROS bag")
    parser.add_argument("bag_file", help="Input ROS bag")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("image_topic", help="Image topic")
    parser.add_argument("pose_topic", help="Pose/Odometry topic")
    
    args = parser.parse_args()
    
    extract_images_and_poses(args.bag_file, args.output_dir, 
                            args.image_topic, args.pose_topic)

if __name__ == '__main__':
    main()