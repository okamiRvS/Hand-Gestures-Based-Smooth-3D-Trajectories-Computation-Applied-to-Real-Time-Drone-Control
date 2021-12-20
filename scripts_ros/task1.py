#!/usr/bin/env python
import rospy
from math import *
from geometry_msgs.msg import Pose2D, Pose
from thymio import ThymioController
from gazebo_msgs.srv import SpawnModel
import pdb

class Task1(ThymioController):

    def __init__(self):
        super(Task1, self).__init__()

        # spawn_model_client = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        # spawn_model_client(
        #     model_name='ground_plane',
        #     model_xml=open('/usr/share/gazebo-9/models/ground_plane/model.sdf', 'r').read(),
        #     robot_namespace='/foo',
        #     initial_pose=Pose(),
        #     reference_frame='world'
        # )


    def takeoff(self):
        # Create a Twist message and add linear x and angular z values
        self.vel_msg.linear.z = .5

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

        step = rospy.Duration.from_sec(3) 

        # For the next 6 seconds publish cmd_vel
        while rospy.Time.now() < now + step:
            rospy.loginfo(f"Current time {rospy.Time.now().secs}s, {rospy.Time.now().nsecs}ns")
            et = now+step
            rospy.loginfo(f"END time {et.secs}s, {et.nsecs}ns")
            self.velocity_publisher.publish(self.vel_msg)
            self.sleep() 

        rospy.loginfo(f"Current time {rospy.Time.now().secs}s, {rospy.Time.now().nsecs}ns")
        et = now+step
        rospy.loginfo(f"END time {et.secs}s, {et.nsecs}ns")
        
        # wait a couple of seconds before execute other things
        now = rospy.Time.now()
        step = rospy.Duration.from_sec(2) 
        self.vel_msg.linear.z = 0
        while rospy.Time.now() < now + step:
            self.velocity_publisher.publish(self.vel_msg)
            self.sleep() 


    def run(self):
        self.sleep()

        # takeoff
        self.takeoff()

        start_time = rospy.Time.now()
        current_pose = None

        while not rospy.is_shutdown():
            elapsed_time = rospy.Time.now() - start_time
            next_time = (elapsed_time + self.step).to_sec()
            goal_pose = self.next_pose(next_time, current_pose)

            if current_pose is not None and next_time<self.dtime[-1]:
                self.vel_msg.linear.x = self.linear_vel(goal_pose.position.x, current_pose.position.x)
                self.vel_msg.linear.y = self.linear_vel(goal_pose.position.y, current_pose.position.y)
                self.vel_msg.linear.z = self.linear_vel(goal_pose.position.z, current_pose.position.z)
                #self.vel_msg.angular.z = self.angular_vel(goal_pose, current_pose)

                #print(self.vel_msg)

                self.velocity_publisher.publish(self.vel_msg)
                self.sleep()
            elif next_time<self.dtime[-1]:
                self.vel_msg.linear.x = 0
                self.vel_msg.linear.y = 0
                self.vel_msg.linear.z = 0
                #self.vel_msg.angular.z = 0
                self.velocity_publisher.publish(self.vel_msg)
                self.sleep()
            else:
                break

            current_pose = goal_pose


    def next_pose(self, time_delta, current_pose):

        # tList = [[0,0,0], [0,0.5,2], [0,2,1.5], [0,2,0], [0,0,0]]
        # dtime = [0, 2.5, 5, 7.5, 10] # nella realtà ci vuole il doppio del tempo
        tList = self.pose
        dtime = self.dtime

        for i, t in enumerate(dtime[1:]):
            
            if t > time_delta:

                # interpolate from two vaules
                idxA = i
                idxB = i+1

                diffTime = dtime[idxB] - dtime[idxA]
                x = tList[idxA][0] + (tList[idxB][0] - tList[idxA][0]) * (time_delta - dtime[idxA]) / (dtime[idxB] - dtime[idxA]) 
                y = tList[idxA][1] + (tList[idxB][1] - tList[idxA][1]) * (time_delta - dtime[idxA]) / (dtime[idxB] - dtime[idxA])
                z = tList[idxA][2] + (tList[idxB][2] - tList[idxA][2]) * (time_delta - dtime[idxA]) / (dtime[idxB] - dtime[idxA])

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
