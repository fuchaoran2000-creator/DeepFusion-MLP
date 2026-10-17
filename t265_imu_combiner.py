#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Imu

latest_gyro = None
latest_gyro_time = None
pub = None

MAX_GYRO_AGE = 0.05  # seconds

def gyro_callback(msg):
    global latest_gyro, latest_gyro_time
    latest_gyro = msg
    latest_gyro_time = msg.header.stamp

def accel_callback(accel_msg):
    global latest_gyro, latest_gyro_time, pub

    if latest_gyro is None:
        return

    dt = abs((accel_msg.header.stamp - latest_gyro_time).to_sec())
    if dt > MAX_GYRO_AGE:
        return

    imu = Imu()
    imu.header.stamp = accel_msg.header.stamp
    imu.header.frame_id = "t265_imu"

    # orientation unknown, do not fake orientation measurement
    imu.orientation.x = 0.0
    imu.orientation.y = 0.0
    imu.orientation.z = 0.0
    imu.orientation.w = 1.0
    imu.orientation_covariance[0] = -1.0

    # real gyro from /t265/gyro/sample
    imu.angular_velocity.x = latest_gyro.angular_velocity.x
    imu.angular_velocity.y = latest_gyro.angular_velocity.y
    imu.angular_velocity.z = latest_gyro.angular_velocity.z
    imu.angular_velocity_covariance = latest_gyro.angular_velocity_covariance

    # real accel from /t265/accel/sample
    imu.linear_acceleration.x = accel_msg.linear_acceleration.x
    imu.linear_acceleration.y = accel_msg.linear_acceleration.y
    imu.linear_acceleration.z = accel_msg.linear_acceleration.z
    imu.linear_acceleration_covariance = accel_msg.linear_acceleration_covariance

    pub.publish(imu)

if __name__ == "__main__":
    rospy.init_node("t265_imu_combiner")

    pub = rospy.Publisher("/imu_combined", Imu, queue_size=200)

    rospy.Subscriber("/t265/gyro/sample", Imu, gyro_callback, queue_size=500)
    rospy.Subscriber("/t265/accel/sample", Imu, accel_callback, queue_size=500)

    rospy.loginfo("Publishing combined T265 IMU on /imu_combined")
    rospy.spin()
