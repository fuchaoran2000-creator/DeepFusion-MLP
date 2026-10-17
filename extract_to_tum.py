#!/usr/bin/env python3
import sys
import rosbag

def stamp_to_sec(stamp):
    return stamp.secs + stamp.nsecs * 1e-9

def write_tum_line(f, stamp, trans, rot):
    t = stamp_to_sec(stamp)
    f.write(
        f"{t:.9f} "
        f"{trans.x:.9f} {trans.y:.9f} {trans.z:.9f} "
        f"{rot.x:.9f} {rot.y:.9f} {rot.z:.9f} {rot.w:.9f}\n"
    )

def extract_pose_stamped(bag_path, topic, out_path):
    count = 0
    with rosbag.Bag(bag_path, "r") as bag, open(out_path, "w") as f:
        for _, msg, _ in bag.read_messages(topics=[topic]):
            write_tum_line(f, msg.header.stamp, msg.pose.position, msg.pose.orientation)
            count += 1
    print(f"Saved {count} PoseStamped poses to {out_path}")

def extract_odom(bag_path, topic, out_path):
    count = 0
    with rosbag.Bag(bag_path, "r") as bag, open(out_path, "w") as f:
        for _, msg, _ in bag.read_messages(topics=[topic]):
            write_tum_line(f, msg.header.stamp, msg.pose.pose.position, msg.pose.pose.orientation)
            count += 1
    print(f"Saved {count} Odometry poses to {out_path}")

def extract_tf(bag_path, topic, parent_frame, child_frame, out_path):
    count = 0
    with rosbag.Bag(bag_path, "r") as bag, open(out_path, "w") as f:
        for _, msg, _ in bag.read_messages(topics=[topic]):
            for tr in msg.transforms:
                if tr.header.frame_id == parent_frame and tr.child_frame_id == child_frame:
                    write_tum_line(
                        f,
                        tr.header.stamp,
                        tr.transform.translation,
                        tr.transform.rotation
                    )
                    count += 1
    print(f"Saved {count} TF poses to {out_path}")

if __name__ == "__main__":
    mode = sys.argv[1]

    if mode == "pose":
        _, _, bag_path, topic, out_path = sys.argv
        extract_pose_stamped(bag_path, topic, out_path)

    elif mode == "odom":
        _, _, bag_path, topic, out_path = sys.argv
        extract_odom(bag_path, topic, out_path)

    elif mode == "tf":
        _, _, bag_path, topic, parent_frame, child_frame, out_path = sys.argv
        extract_tf(bag_path, topic, parent_frame, child_frame, out_path)

    else:
        print("Unknown mode")
        sys.exit(1)
