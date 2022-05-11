# Hand Gestures-Based Smooth 3D Trajectories Computation Applied to Real-Time Drone Control by Tracking 2D Hand Landmarks

## Abstract
Robotic systems are increasingly being adopted for flming and photography purposes. In fact, by exploiting robots, it is possible to program complex motions, achieving high-quality videos and photographs. The contribution of the presented paper is an alternative approach for robot motion planning w.r.t. the joystick. To this end, a Deep Neural Network which recognizes the gestures of a hand is constructed. Then, a pipeline reconstructing 3D trajectories obtained from 2D reference points is proposed. Finally, 3D movements using a
state-of-the-art hand tracking system can be acquired, estimating the orientation of the hand and its depth position as well. 3D trajectories are interpolated and smoothed with Ridge
Regression. To evaluate the proposed remote control approach, a captured trajectory is tested in a simulation environment to control the motion of a drone. In addition, experiments are
provided using the DJI Ryze Tello drone to prove the feasibility of the approach in real conditions.


## Setup
* `pip install -r requirements.txt`
* Open `scripts` folder in visual studio code, or set it as main directory

## Acknowledgements
Working on this thesis was an experience that enriched me. I was able to work on a big
project that allowed me to put into practice most of the skills acquired in these years of
university. I understood how complicated it is to take small steps forward in any area
of knowledge day after day.

I am glad I worked on this thesis and I could not have asked for better regarding the
support of my advisors: Dr. Alessandro Giusti and Dr. Loris Roveda. They consistently allowed this paper to be my own work but steered me in the right direction whenever they thought I needed it.