from rosbags.highlevel import AnyReader
from pathlib import Path

bag_path = Path.home() / 'data' / 'indoor_forward_5_snapdragon_with_gt.bag'

with AnyReader([bag_path]) as reader:
    print(f"Duration: {reader.duration / 1e9:.2f} seconds")
    
    for connection, timestamp, rawdata in reader.messages():
        if connection.topic == '/snappy_imu':
            msg = reader.deserialize(rawdata, connection.msgtype)

            print(f"Timestamp: {timestamp}")
            print(f"Header")
            print(f"Seq: {msg.header.seq}")
            print(f"Stamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
            print(f"Frame id: {msg.header.frame_id}")
            print(f"\nangular_velocity [rad/sec]:")
            print(f"  - x: {msg.angular_velocity.x}")
            print(f"  - y: {msg.angular_velocity.y}")
            print(f"  - z: {msg.angular_velocity.z}")
            print(f"\nlinear_acceleration [m/s^2]:")
            print(f"  - x: {msg.linear_acceleration.x}")
            print(f"  - y: {msg.linear_acceleration.y}")
            print(f"  - z: {msg.linear_acceleration.z}")
            print(f"\norientation:")
            print(f"  - x: {msg.orientation.x}")
            print(f"  - y: {msg.orientation.y}")
            print(f"  - z: {msg.orientation.z}")
            print(f"  - w: {msg.orientation.w}")
    
    for connection, timestamp, rawdata in reader.messages():
        if connection.topic == '/groundtruth/pose':
            msg = reader.deserialize(rawdata, connection.msgtype)

            # Check if values are not NaN
            position = msg.pose.position
            orientation = msg.pose.orientation

            # Check if any values are meaningful (not NaN)
            import math
            has_valid_position = not (math.isnan(position.x) or math.isnan(position.y) or math.isnan(position.z))
            has_valid_orientation = not (math.isnan(orientation.x) or math.isnan(orientation.y) or math.isnan(orientation.z) or math.isnan(orientation.w))

            if has_valid_position or has_valid_orientation:
                print(f"\n=== Valid data found ===")
                print(f"Timestamp: {timestamp}")
                print(f"Header")
                print(f"Seq: {msg.header.seq}")
                print(f"Stamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
                print(f"Frame id: {msg.header.frame_id}")
                print(f"\nPosition:")
                print(f"  - x: {position.x}")
                print(f"  - y: {position.y}")
                print(f"  - z: {position.z}")
                print(f"\nOrientation:")
                print(f"  - x: {orientation.x}")
                print(f"  - y: {orientation.y}")
                print(f"  - z: {orientation.z}")
                print(f"  - w: {orientation.w}")

                # Also print twist if available
                if hasattr(msg, 'twist'):
                    linear_vel = msg.twist.twist.linear
                    angular_vel = msg.twist.twist.angular
                    print(f"\nLinear Velocity:")
                    print(f"  - x: {linear_vel.x}")
                    print(f"  - y: {linear_vel.y}")
                    print(f"  - z: {linear_vel.z}")
                    print(f"\nAngular Velocity:")
                    print(f"  - x: {angular_vel.x}")
                    print(f"  - y: {angular_vel.y}")
                    print(f"  - z: {angular_vel.z}")
                print("="*30)

    # Print image information
    print("\n\n=== Camera Image Information ===")
    image_count = 0
    for connection, timestamp, rawdata in reader.messages():
        if connection.topic == '/snappy_cam/stereo_l':
            msg = reader.deserialize(rawdata, connection.msgtype)
            image_count += 1

            print(f"\nImage #{image_count}")
            print(f"Timestamp: {timestamp}")
            print(f"Header:")
            print(f"  Seq: {msg.header.seq}")
            print(f"  Stamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
            print(f"  Frame id: {msg.header.frame_id}")
            print(f"Image:")
            print(f"  Height: {msg.height} pixels")
            print(f"  Width: {msg.width} pixels")
            print(f"  Encoding: {msg.encoding}")
            print(f"  Is bigendian: {msg.is_bigendian}")
            print(f"  Step: {msg.step} bytes")
            print(f"  Data size: {len(msg.data)} bytes")
            print("-"*30)

    print(f"\nTotal images: {image_count}")
