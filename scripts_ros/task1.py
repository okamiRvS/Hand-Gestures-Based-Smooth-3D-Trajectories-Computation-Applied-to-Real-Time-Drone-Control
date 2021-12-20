#!/usr/bin/env python
import rospy
from math import *
from geometry_msgs.msg import Pose2D, Pose
from thymio import ThymioController
import pdb

class Task1(ThymioController):

    def __init__(self):
        super(Task1, self).__init__()

        self.R, self.T = 1, 30 #radius, period

    def run(self):
        self.sleep()

        start_time = rospy.Time.now()
        current_pose = None

        while not rospy.is_shutdown():
            elapsed_time = rospy.Time.now() - start_time
            next_time = (elapsed_time + self.step).to_sec()
            goal_pose = self.next_pose(next_time, current_pose)

            if current_pose is not None and next_time<10:
                self.vel_msg.linear.x = self.linear_vel(goal_pose.position.x, current_pose.position.x)
                self.vel_msg.linear.y = self.linear_vel(goal_pose.position.y, current_pose.position.y)
                self.vel_msg.linear.z = self.linear_vel(goal_pose.position.z, current_pose.position.z)
                #self.vel_msg.angular.z = self.angular_vel(goal_pose, current_pose)

                #print(self.vel_msg)

                self.velocity_publisher.publish(self.vel_msg)
                self.sleep()
            else:
                self.vel_msg.linear.x = 0
                self.vel_msg.linear.y = 0
                self.vel_msg.linear.z = 0
                #self.vel_msg.angular.z = 0
                self.velocity_publisher.publish(self.vel_msg)
                self.sleep()
                print("vel is 0") 

            current_pose = goal_pose

    def next_pose(self, time_delta, current_pose):

        tList = [[0,0,0], [0,0.5,2], [0,2,1.5], [0,2,0], [0,0,0]]
        time = [0, 2.5, 5, 7.5, 10] # nella realtà ci vuole il doppio del tempo

        for i, t in enumerate(time[1:]):
            
            if t > time_delta:

                # interpolate from two vaules
                idxA = i
                idxB = i+1

                diffTime = time[idxB] - time[idxA]
                x = tList[idxA][0] + (tList[idxB][0] - tList[idxA][0]) * (time_delta - time[idxA]) / (time[idxB] - time[idxA]) 
                y = tList[idxA][1] + (tList[idxB][1] - tList[idxA][1]) * (time_delta - time[idxA]) / (time[idxB] - time[idxA])
                z = tList[idxA][2] + (tList[idxB][2] - tList[idxA][2]) * (time_delta - time[idxA]) / (time[idxB] - time[idxA])

                theta = 0

                #pose = Pose2D(x, y, theta)
                pose = Pose()
                pose.position.x = x
                pose.position.y = y
                pose.position.z = z
                print(x, "\t", y, "\t", z)

                break

            else:
                pose = None
            
        return pose


if __name__ == '__main__':
    try:
        controller = Task1()
        controller.run()
    except rospy.ROSInterruptException as e:
        pass
