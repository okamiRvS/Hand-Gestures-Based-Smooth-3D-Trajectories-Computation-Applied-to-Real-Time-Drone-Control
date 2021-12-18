#!/usr/bin/env python
from inspect import trace
import rospy
from math import *
import tf
from geometry_msgs.msg import Pose2D, Twist
from nav_msgs.msg import Odometry
import pdb


def main():

    rospy.init_node('controllerByScript', anonymous=True)

    # Create a publisher which can "talk" to Turtlesim and tell it to move
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
     
    # Create a Twist message and add linear x and angular z values
    move_cmd = Twist()
    move_cmd.linear.x = 1.0
    move_cmd.angular.z = 1.0

    #https://roboticsbackend.com/ros-rate-roscpy-roscpp/
    # Save current time and set publish rate at 20 Hz
    now = rospy.Time.now()

    # When using simulated Clock time, rospy.Time.now() returns time 0 until first message has been 
    # received on /clock, so 0 means essentially that the client does not know clock time yet. 
    # A value of 0 should therefore be treated differently, such as looping over rospy.Time.now()
    # until non-zero is returned.
    while now.secs == 0:
        now = rospy.Time.now()
        #print("NON ANCORA")

    frequency = 20.0
    rate = rospy.Rate(frequency)
    step = rospy.Duration.from_sec(2) 

    # For the next 6 seconds publish cmd_vel
    while rospy.Time.now() < now + step:
        rospy.loginfo(f"Current time {rospy.Time.now().secs}s, {rospy.Time.now().nsecs}ns")
        et = now+step
        rospy.loginfo(f"END time {et.secs}s, {et.nsecs}ns")
        pub.publish(move_cmd)
        rate.sleep() 

    rospy.loginfo(f"Current time {rospy.Time.now().secs}s, {rospy.Time.now().nsecs}ns")
    et = now+step
    rospy.loginfo(f"END time {et.secs}s, {et.nsecs}ns")

if __name__ == '__main__':
    main()


#rostopic pub -r 10 /cmd_vel geometry_msgs/Twist  '{linear:  {x: 0.1, y: 0.0, z: 0.0}, angular: {x: 0.0,y: 0.0,z: 0.0}}'