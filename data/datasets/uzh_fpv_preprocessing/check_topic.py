from rosbags.rosbag1 import Reader
import argparse

def print_topic(bag_file):
    with Reader(bag_file) as r:
        for c in r.connections:
            print(f'{c.topic} / {c.msgtype} / {c.msgcount}')



def main():
    parser = argparse.ArgumentParser(description="Check topic")
    parser.add_argument("bag_file", help="Input ROS bag")
    args = parser.parse_args()
    
    print_topic(args.bag_file)

if __name__=='__main__':
    main()