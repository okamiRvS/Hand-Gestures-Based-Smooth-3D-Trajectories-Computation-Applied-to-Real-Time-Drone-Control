#!/usr/bin/env python
import sys
sys.path.append('/home/usiusi/catkin_ws/src/DJI-Tello-3D-Hand-Gesture-control/scripts')
import fullControllModule

import rospy
from math import *
from geometry_msgs.msg import Pose2D, Pose, Twist
from thymio import ThymioController, PID
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SpawnModel, DeleteModel, GetModelState, SetModelState
import pdb


class Task1(ThymioController):

    def __init__(self):
        super(Task1, self).__init__()

        # to get position of a scene object
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            self.get_model_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

        # to spawn an object
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            self.spawn_model_client = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

        self.numberSpheres = -1

        self.time_take_off = 5


    def drawPoint(self, event): # callback

        print('Timer called at ' + str(event.current_real))

        # https://www.youtube.com/watch?v=WqK2IY5_9OQ
        # the second value means took this obj, not child as reference
        try:
            current_pose = self.get_model_state("quadrotor", "")

            self.numberSpheres += 1

            self.spawn_model_client(
                model_name = f"ball-{self.numberSpheres}",
                model_xml = open('/home/usiusi/catkin_ws/src/tello_ros_gazebo/tello_driver/models/my1stmodel/model.sdf', 'r').read(),
                robot_namespace = '',
                initial_pose = current_pose.pose,
                reference_frame = 'world'
            )
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)


    def takeoff(self):
        # # Create a Twist message and add linear x and angular z values
        # self.vel_msg.linear.z = .3

        # #https://roboticsbackend.com/ros-rate-roscpy-roscpp/
        # now = rospy.Time.now()

        # # When using simulated Clock time, rospy.Time.now() returns time 0 until first message has been 
        # # received on /clock, so 0 means essentially that the client does not know clock time yet. 
        # # A value of 0 should therefore be treated differently, such as looping over rospy.Time.now()
        # # until non-zero is returned.
        # while now.secs == 0:
        #     now = rospy.Time.now()
        #     #print("NON ANCORA")

        # step = rospy.Duration.from_sec(3) 

        # # For the next 6 seconds publish cmd_vel
        # while rospy.Time.now() < now + step:
        #     rospy.loginfo(f"Current time {rospy.Time.now().secs}s, {rospy.Time.now().nsecs}ns")
        #     et = now+step
        #     rospy.loginfo(f"END time {et.secs}s, {et.nsecs}ns")
        #     self.velocity_publisher.publish(self.vel_msg)
        #     self.sleep() 

        # rospy.loginfo(f"Current time {rospy.Time.now().secs}s, {rospy.Time.now().nsecs}ns")
        # et = now+step
        # rospy.loginfo(f"END time {et.secs}s, {et.nsecs}ns")
        

        # When using simulated Clock time, rospy.Time.now() returns time 0 until first message has been 
        # received on /clock, so 0 means essentially that the client does not know clock time yet. 
        # A value of 0 should therefore be treated differently, such as looping over rospy.Time.now()
        # until non-zero is returned.
        start_time = rospy.Time.now()
        while start_time.secs == 0:
            start_time = rospy.Time.now()

        current_pose = self.get_model_state("quadrotor", "").pose

        # x_controll = PID(1, 0, 1)
        # y_controll = PID(1, 0, 1)
        # x_ang_controll = PID(1, 0, 1)
        # y_ang_controll = PID(1, 0, 1)
        # z_ang_controll = PID(1, 0, 1)

        while not rospy.is_shutdown():
            elapsed_time = rospy.Time.now() - start_time
            next_time = (elapsed_time + self.step).to_sec()
            goal_pose = self.next_takeoff_pose(next_time)

            # pos_drone = self.get_model_state("quadrotor", "").pose
            # self.vel_msg.linear.x = x_controll.step(pos_drone.position.x, self.step.to_sec())
            # self.vel_msg.linear.y = y_controll.step(pos_drone.position.y, self.step.to_sec())
            # self.vel_msg.angular.x = x_ang_controll.step(pos_drone.orientation.x, self.step.to_sec())
            # self.vel_msg.angular.y = y_ang_controll.step(pos_drone.orientation.y, self.step.to_sec())
            # self.vel_msg.angular.z = z_ang_controll.step(pos_drone.orientation.x, self.step.to_sec())
            
            if next_time < self.time_take_off + 2:
                self.vel_msg.linear.z = self.linear_vel(goal_pose.position.z, current_pose.position.z)
                self.velocity_publisher.publish(self.vel_msg) 
            else:
                self.vel_msg.linear.z = 0
                self.velocity_publisher.publish(self.vel_msg)
                self.sleep()
                break

            current_pose = goal_pose
            self.sleep()

            
    def next_takeoff_pose(self, time_delta):
            
        pose = Pose()
        if self.time_take_off > time_delta:

            # we have to start not from zero, otherwise there are problems
            z = 0.1 + 3 * time_delta / self.time_take_off

            pose.position.z = z

        quadr = self.get_model_state("quadrotor", "")
        print(time_delta)
        print(quadr.pose)
            
        return pose


    def set_position(self, for_x_time, state_msg):

        start_time = rospy.Time.now()
        while start_time.secs == 0:
            start_time = rospy.Time.now()

        while not rospy.is_shutdown():
            elapsed_time = (rospy.Time.now() - start_time).to_sec()

            if elapsed_time < for_x_time:

                rospy.wait_for_service('/gazebo/set_model_state')
                try:
                    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
                    resp = set_state( state_msg )

                except rospy.ServiceException as e:
                    print("Service call failed: %s" % e)

                self.vel_msg = Twist()
                self.velocity_publisher.publish(self.vel_msg)

                #pos_drone = self.get_model_state("quadrotor", "").pose
                #print(pos_drone)
            else:
                break
            
            self.sleep() 

    def test2(self):

        # set position and orientation of the drone
        state_msg = ModelState()
        state_msg.model_name = 'quadrotor'

        print("Initialization...")
        state_msg.pose.position.z = 0.03
        self.set_position(2, state_msg)

        self.takeoff() #### fly fly little drone

        print("Wait a bit before execute trajectory...")
        pos_drone = self.get_model_state("quadrotor", "").pose
        state_msg.pose.position.z = pos_drone.position.z
        self.set_position(1, state_msg)

    def run(self):

        #self.resetPose() # it doesn't work this...

        # set position and orientation of the drone
        state_msg = ModelState()
        state_msg.model_name = 'quadrotor'

        print("Initialization...")
        state_msg.pose.position.z = 0.03
        self.set_position(2, state_msg)

        self.takeoff() #### fly fly little drone

        print("Wait a bit before execute trajectory...")
        pos_drone = self.get_model_state("quadrotor", "").pose
        state_msg.pose.position.z = pos_drone.position.z
        self.set_position(1, state_msg)

        # start to execute the trajectory
        start_time = rospy.Time.now()
        current_pose = None

        # istantiate the sphere in the scene each n seconds
        callBack = rospy.Timer(rospy.Duration(0.2), self.drawPoint)

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

                # black timer
                callBack.shutdown()

                # delete all sphere
                if self.numberSpheres >= 0:

                    for i in range(self.numberSpheres):
                        try:
                            flag = True
                            while(flag): # exit only if obj deleted
                                delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
                                result = delete_model(f"ball-{i}")
                                if result.success == True:
                                    print(f"{result.status_message}: ball-{i}")
                                    flag = False
                        except rospy.ServiceException as e:
                            print("Service call failed: %s"%e)

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

                pose = Pose()
                pose.position.x = x
                pose.position.y = y
                pose.position.z = z
                #print(x, "\t", y, "\t", z)

                break
            else:
                pose = None
            
        return pose


if __name__ == '__main__':

    try:
        controller = Task1()

        fullControll = fullControllModule.FullControll()
        
        while True:
            #Get data from hand
            resTraj = fullControll.main()
            controller.normalizeData(resTraj)

            # execute trajectory
            controller.run()
            
            # test
            #controller.test2()

    except rospy.ROSInterruptException as e:
        pass
