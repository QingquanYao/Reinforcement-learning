import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import os

class DoublePendulumEnv(gym.Env):
    """
    Description:
        A pole is attached to one of the ends of another pole, whoes the other 
        end is attached to a fixed axle. The pendulum starts downright, and the
        goal is to stand upside down by increasing and reducing the poles' angular
        velocity.

    Source:
        This environment corresponds to

    Observation:
        Type: Box(4)
        Num     Observation                 Min                     Max
        0       Pole_1 Angle               -pi rad (-180 deg)       pi rad (180 deg)    
        1       Pole_2 Angle               -pi rad (-180 deg)       pi rad (180 deg)
        2       Pole_2 Angular Velocity    -Inf                     Inf
        3       Pole_2 Angular Velocity    -Inf                     Inf

Continuous controll:
    Actions:
        Type: Box(1)
        Num     Action                   Min                     Max
        0       External torque          -2.0                    2.0
                acting on Pole_2 


    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push pole to the clockwise
        1     Push pole to the anti-clockwise

    Reward(????):
        Reward is 1 for two poles inverted upright 

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination(???):
        Episode length is greater than 200.
    """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, g=10.0):
        self.name=os.path.splitext(os.path.basename(__file__))[0]
        self.max_speed = 12  
        self.max_speed_1= self.max_speed 
        self.max_speed_2=self.max_speed 

        self.max_force = 1.0

        self.dt = 0.05
        self.g = g

        self.m1 = 5.0 # mass of Pole_1
        self.m2 = 1.0 # mass of Pole_2

        self.l1 = 1.5 # length of Pole_1
        self.l2 = 1.0 # length of Pole_2

        self.theta_1=np.pi
        self.theta_2=np.pi
        
        self.viewer = None

        # high = np.array([self.l1/2, self.l1/2, self.max_speed, self.l1+self.l2/2 , self.l1+self.l2/2 , self.max_speed], dtype=np.float32)
        # # the above is the up bound of observations [maximum position of mass ceneter of Pole_i on x direction, on y direction, maximum angular velocity of Pole_i]
        # ## or another way to rewrit observations [angle of Pole_i, angular velocity of Pole_i]
        # high = np.array([self.theta_1, self.max_speed_1, self.theta_2, self.max_speed_2], dtype=np.float32)
        # self.action_space = spaces.Box(
        #     low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        # )
        # self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # self.seed()

        high = np.array([1.0, 1.0,1.0, 1.0, self.max_speed_1, self.max_speed_2], dtype=np.float32)
        # self.action_space = spaces.Box(
        #     low=-self.max_force, high=self.max_force, shape=(1,), dtype=np.float32
        # )
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, F):
        theta_1, theta_2, thdot1, thdot2 = self.state  # th := theta
        assert self.action_space.contains(F)
        g = self.g
        m_1 = self.m1
        m_2 = self.m2
        l_1 = self.l1
        l_2 = self.l2
        dt = self.dt
        
        F = self.max_force if F == 1 else -self.max_force
        # F = np.clip(F, -self.max_force, self.max_force)[0]
        self.last_F = F  # for rendering
        costs = angle_normalize(theta_1) ** 2 + angle_normalize(theta_2) ** 2 +0.1 * thdot1 ** 2 + 0.1 * thdot2 ** 2 +0.001 * (F ** 2)

        newthdot1 = thdot1 + ((12*F *np.sin(theta_1  - np.arcsin(l_2 *np.sin(theta_1  - theta_2 )/np.sqrt(l_1**2 + 2*l_1*l_2 *np.cos(theta_1  - theta_2 ) + l_2**2))) *np.sin(theta_1 ) - 18*F *np.sin(theta_1  - np.arcsin(l_2 *np.sin(theta_1  - theta_2 )/np.sqrt(l_1**2 + 2*l_1*l_2 *np.cos(theta_1  - theta_2 ) + l_2**2))) *np.sin(theta_2 ) *np.cos(theta_1  - theta_2 ) - 18*F *np.cos(theta_1  - theta_2 ) *np.cos(theta_1  - np.arcsin(l_2 *np.sin(theta_1  - theta_2 )/np.sqrt(l_1**2 + 2*l_1*l_2 *np.cos(theta_1  - theta_2 ) + l_2**2))) *np.cos(theta_2 ) + 12*F *np.cos(theta_1  - np.arcsin(l_2 *np.sin(theta_1  - theta_2 )/np.sqrt(l_1**2 + 2*l_1*l_2 *np.cos(theta_1  - theta_2 ) + l_2**2))) *np.cos(theta_1 ) - 6*g*m_1 *np.sin(theta_1 ) - 12*g*m_2 *np.sin(theta_1 ) + 9*g*m_2 *np.sin(theta_2 ) *np.cos(theta_1  - theta_2 ) - 9*l_1*m_2 *np.sin(2*theta_1  - 2*theta_2 )*thdot1**2/2 - 6*l_2*m_2 *np.sin(theta_1  - theta_2 )*thdot2**2)/(l_1*(3*m_1 - 9*m_2 *np.cos(theta_1  - theta_2 )**2 + 13*m_2))) * dt
        newthdot1 = np.clip(newthdot1, -self.max_speed, self.max_speed)
        newth1 = theta_1 + newthdot1 * dt

        newthdot2 = thdot2 + (3*(6*F*m_1 *np.sin(theta_1  - np.arcsin(l_2 *np.sin(theta_1  - theta_2 )/np.sqrt(l_1**2 + 2*l_1*l_2 *np.cos(theta_1  - theta_2 ) + l_2**2))) *np.sin(theta_2 ) + 6*F*m_1 *np.cos(theta_1  - np.arcsin(l_2 *np.sin(theta_1  - theta_2 )/np.sqrt(l_1**2 + 2*l_1*l_2 *np.cos(theta_1  - theta_2 ) + l_2**2))) *np.cos(theta_2 ) - 12*F*m_2 *np.sin(theta_1  - np.arcsin(l_2 *np.sin(theta_1  - theta_2 )/np.sqrt(l_1**2 + 2*l_1*l_2 *np.cos(theta_1  - theta_2 ) + l_2**2))) *np.sin(theta_1 ) *np.cos(theta_1  - theta_2 ) + 26*F*m_2 *np.sin(theta_1  - np.arcsin(l_2 *np.sin(theta_1  - theta_2 )/np.sqrt(l_1**2 + 2*l_1*l_2 *np.cos(theta_1  - theta_2 ) + l_2**2))) *np.sin(theta_2 ) - 12*F*m_2 *np.cos(theta_1  - theta_2 ) *np.cos(theta_1  - np.arcsin(l_2 *np.sin(theta_1  - theta_2 )/np.sqrt(l_1**2 + 2*l_1*l_2 *np.cos(theta_1  - theta_2 ) + l_2**2))) *np.cos(theta_1 ) + 26*F*m_2 *np.cos(theta_1  - np.arcsin(l_2 *np.sin(theta_1  - theta_2 )/np.sqrt(l_1**2 + 2*l_1*l_2 *np.cos(theta_1  - theta_2 ) + l_2**2))) *np.cos(theta_2 ) + 6*g*m_1*m_2 *np.sin(theta_1 ) *np.cos(theta_1  - theta_2 ) - 3*g*m_1*m_2 *np.sin(theta_2 ) + 12*g*m_2**2 *np.sin(theta_1 ) *np.cos(theta_1  - theta_2 ) - 13*g*m_2**2 *np.sin(theta_2 ) + 3*l_1*m_1*m_2 *np.sin(theta_1  - theta_2 )*thdot1**2 + 13*l_1*m_2**2 *np.sin(theta_1  - theta_2 )*thdot1**2 + 3*l_2*m_2**2 *np.sin(2*theta_1  - 2*theta_2 )*thdot2**2)/(2*l_2*m_2*(3*m_1 - 9*m_2 *np.cos(theta_1  - theta_2 )**2 + 13*m_2))) * dt
        newthdot2 = np.clip(newthdot2, -self.max_speed, self.max_speed)
        newth2 = theta_2 + newthdot2 * dt

        self.state = np.array([newth1, newth2, newthdot1, newthdot2])
        return self._get_obs(), -costs, False, {}

    def reset(self):
        high = np.array([2*np.pi,2*np.pi,1, 1]) #[theta_1 theta_2 theta_dot_1 theta_dot_2]
        self.state = self.np_random.uniform(low=-high, high=high)
        # self.state = np.array([np.pi,np.pi,1, 1]) #droping from upright 
        self.last_F = None
        return self._get_obs()

    def _get_obs(self):
        l1 = self.l1
        l2 = self.l2
        the1, the2, thetadot_1, thetadot_2 = self.state
        x1=l1/2*np.sin(the1)
        y1 = -l1/2*np.cos(the1)
        x2 = l1*np.sin(the1)+l2/2*np.sin(the2)
        y2 = -l1*np.cos(the1)-l2/2*np.cos(the2)
        return np.array([x1, y1, x2, y2, thetadot_1, thetadot_2], dtype=np.float32)

    def render(self, mode="human"):
       
        if self.viewer is None:
            from our_environments.envs import rendering
            self.viewer = rendering.Viewer(600, 600)
            self.viewer.set_bounds(-4.2, 4.2, -4.2, 4.2)
            rod_1 = rendering.make_capsule(self.l1, 0.05)
            rod_1.set_color(0.8, 0.3, 0.3)
            self.pole1_transform = rendering.Transform(translation=(0.0, 0.0))
            #translation=(0.0,0.0) controll the position of transformation center
            rod_1.add_attr(self.pole1_transform)
            self.viewer.add_geom(rod_1)

            rod_2 = rendering.make_capsule(self.l2, 0.05)
            rod_2.set_color(0.8, 0.3, 0.3)
            self.pole2_transform = rendering.Transform()
            #translation=(0.0,0.0) controll the position of transformation center
            rod_2.add_attr(self.pole2_transform)
            self.viewer.add_geom(rod_2)

            joint = rendering.make_circle(0.03)
            joint.set_color(0, 0, 0)
            self.joint_transform = rendering.Transform() #translation=(0.0,0.0) controll the position of transformation center
            joint.add_attr(self.joint_transform)   
            self.viewer.add_geom(joint)

            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)

            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1.0, 1.0)
            self.imgtrans = rendering.Transform(translation=(0, 0))
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole1_transform.set_rotation(self.state[0]-np.pi/2)
        self.pole2_transform.set_rotation(self.state[1]-np.pi/2)
        self.joint_transform.set_translation(self.l1*np.cos(self.state[0]-np.pi/2), self.l1*np.sin(self.state[0]-np.pi/2)) 
        self.pole2_transform.set_translation(self.l1*np.cos(self.state[0]-np.pi/2), self.l1*np.sin(self.state[0]-np.pi/2))      
        if self.last_F is not None:
            self.imgtrans.scale = (-self.last_F / 2, np.abs(self.last_F) / 2)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
