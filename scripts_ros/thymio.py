#!/usr/bin/env python
from os import read
import rospy
from math import *
import tf
from geometry_msgs.msg import Pose2D, Twist
from hector_uav_msgs.srv import EnableMotors
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
import pdb
import csv
import numpy as np


class ThymioController(object):

    def __init__(self):

        rospy.init_node('controllerByScript', anonymous=True)
        self.velocity_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.pose_subscriber = rospy.Subscriber(f"/tello/odom", Odometry, self.log_odometry)
        
        # we need enable the motors to move the drone in the simulation
        rospy.wait_for_service("/enable_motors") # wait until service isn't running
        fan = rospy.ServiceProxy("/enable_motors", EnableMotors)
        fan(enable=True)

        self.pose2d = Pose2D()
        self.vel_msg = Twist()
        rospy.on_shutdown(self.stop)
        frequency = 20.0
        self.rate = rospy.Rate(frequency)
        self.step = rospy.Duration.from_sec(1.0 / frequency)

        # reset model poses when we launch the script
        rospy.wait_for_service("/gazebo/reset_world")
        reset = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        reset()
        self.sleep()

        self.readCsv() # test

        
    def readCsv(self):
        self.pose = []
        self.orientation = []
        self.dtime = []
        self.speed = []
        with open('/home/usiusi/catkin_ws/src/tello_ros_gazebo/tello_driver/scripts/data.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                #print(f'{", ".join(row)}')
                #print("\n\n")
                
                elem = [float(x) for x in row]

                if line_count < 3:
                    self.pose.append(elem)
                elif line_count < 6:
                    self.orientation.append(elem)
                elif line_count == 6:
                    self.dtime.append(elem)
                elif line_count == 7:
                    self.speed.append(elem)
                
                line_count+=1

        self.pose = np.vstack((self.pose[1], self.pose[0] , self.pose[2])).T * 5
        #print(self.pose)
        self.orientation = np.vstack((self.orientation[0], self.orientation[1], self.orientation[2])).T
        self.dtime = self.dtime[0]
        self.speed = self.speed[0]


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
        self.pose2d = self.human_readable_pose2d(data.pose.pose)

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