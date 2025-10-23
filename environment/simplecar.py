'''
Contains car object based on kinematic bicycle model
'''
import numpy as np

class SimpleCar:
    def __init__(self, init_pos, init_yaw=0.0, wheelbase=2.5, dt=0.1, max_speed=50.0, max_steer = np.deg2rad(30), accel_rate = 10.0, brake_rate= 20.0, c_drag=0.0025, c_roll=0.015, k_slip=0.02):
        '''
        Basic kinematic car model.
        Args:
            init_pos (tuple): (x, y) start position.
            init_yaw (float): initial heading in radians.
            wheelbase (float): distance between front and rear axles.
            dt (float): timestep for updates.
            max_speed (float): maximum forward speed (m/s).
            max_steer (float): max steering angle (radians).
            accel_rate (float): acceleration per second at full throttle.
            brake_rate (float): deceleration per second at full brake.
            c_drag: drag coeff
            c_roll: rolling resistance coeff
            k_slip: lateral grip coeff
        '''

        self.x, self.y = init_pos
        self.yaw = init_yaw
        self.v = 0.0

        # constants -> car configurations
        self.dt = dt
        self.L = wheelbase
        self.max_speed = max_speed
        self.max_steer = max_steer
        self.accel_rate = accel_rate
        self.brake_rate = brake_rate
        self.c_drag = c_drag 
        self.c_roll = c_roll
        self.k_slip = k_slip

    def reset(self, pos, yaw=0.0):
        '''
        Reset car to a specific position
        '''
        self.x, self.y = pos
        self.yaw = yaw
        self.v = 0.0

    def step(self, throttle, steer):
        '''
        Update the car state given throttle and steering.
        throttle: [-1, 1] (negative = brake)
        steer: [-1, 1] (normalized steering)
        '''
        steer = np.clip(steer, -1.0, 1.0) * self.max_steer
        throttle = np.clip(throttle, -1.0, 1.0)

        # Acceleration
        if throttle >= 0:
            a = throttle * self.accel_rate
        else: a = throttle * self.brake_rate

        # apply resistances
        a_resist = -self.c_roll - self.c_drag * self.v**2
        a_total = a + a_resist

        # apply grip limit
        steer_eff = steer / (1 + self.k_slip * self.v**2)

        # Motion from kinematic equations
        self.x += self.v * np.cos(self.yaw) * self.dt
        self.y += self.v * np.sin(self.yaw) * self.dt
        self.yaw += (self.v / self.L) * np.tan(steer_eff) * self.dt
        self.v += a_total* self.dt
        self.v = np.clip(self.v, 0, self.max_speed)

        # Return updated state
        return np.array([self.x, self.y, self.v, self.yaw]) 


    def get_state(self):
        '''
        Return current state as dict
        '''
        return {'x': self.x, 'y': self.y, 'yaw': self.yaw, 'v': self.v}
    