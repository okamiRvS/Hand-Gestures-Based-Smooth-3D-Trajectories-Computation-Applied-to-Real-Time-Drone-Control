#!/usr/bin/env python
import rospy
from math import *
import tf
from geometry_msgs.msg import Pose2D, Twist
from nav_msgs.msg import Odometry
import pdb


class ThymioController(object):

    def __init__(self):

        rospy.init_node('controllerByScript', anonymous=True)
        self.velocity_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.pose_subscriber = rospy.Subscriber(f"/tello/odom", Odometry, self.log_odometry)
        self.pose = Pose2D()
        self.vel_msg = Twist()
        rospy.on_shutdown(self.stop)
        frequency = 20.0
        self.rate = rospy.Rate(frequency)
        self.step = rospy.Duration.from_sec(1.0 / frequency)


    def human_readable_pose2d(self, pose):

        #Return pose2D
        quaternion = (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        )
        roll, pitch, yaw = tf.transformations.euler_from_quaternion(quaternion)

        return Pose2D(pose.position.x, pose.position.y, yaw)


    def log_odometry(self, data):

        #Pose and velocities update
        self.vel_msg = data.twist.twist
        self.pose = self.human_readable_pose2d(data.pose.pose)

        # log robot's pose
        rospy.loginfo_throttle(
            period=2,  # log every 10 seconds
            msg= f"{self.pose.x}, {self.pose.y}, {self.pose.theta}"
        )

    def euclidean_distance(self, goal_pose, current_pose):

        #Return Euclidean distance between current pose and the goal pose

        return goal_pose - current_pose

    def linear_vel(self, goal_pose, current_pose):

        #Return clipped linear velocity
        distance = self.euclidean_distance(goal_pose, current_pose)
        velocity = distance / self.step.to_sec()

        return velocity


    def angular_difference(self, current_pose, goal_pose):

        #Return delta angle
        delta_ang = goal_pose.theta - current_pose.theta

        return atan2(sin(delta_ang), cos(delta_ang))


    def angular_vel(self, goal_pose, current_pose):

        #Return the angular velocity using the delta angle
        delta_ang = self.angular_difference(current_pose, goal_pose)
        velocity = delta_ang / self.step.to_sec()

        return velocity


    def sleep(self):
        self.rate.sleep()
        if rospy.is_shutdown():
            raise rospy.ROSInterruptException


    def stop(self):
        self.velocity_publisher.publish(Twist())
        self.sleep()


class PID:
    def __init__(self, Kp, Ki, Kd, min_out=-float("inf"), max_out=float("inf")):
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki

        self.min_out = min_out
        self.max_out = max_out

        self.last_e = None
        self.sum_e = 0

    def step(self, e, dt):
        #dt is the time elapsed from the last time step was called
        if self.last_e is not None:
            derivative = (e - self.last_e) / dt
        else:
            derivative = 0

        self.last_e = e
        self.sum_e += e * dt

        output = self.Kp * e + self.Kd * derivative + self.Ki * self.sum_e 
        return min(max(self.min_out, output), self.max_out)