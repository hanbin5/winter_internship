from rosbags.rosbag1 import Reader

with Reader('/home/hanbin5/data/UZH-FPV/race_2/race_2.bag') as r:
    for c in r.connections:
        print(f'{c.topic} / {c.msgtype} / {c.msgcount}')