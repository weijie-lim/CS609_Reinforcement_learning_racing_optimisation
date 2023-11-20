"""
Classes:

Car: Agent
FixedTrack: Allows setting of radius, random weather
RandomizedTrack: Random radius, random weather
EvaluationTrack: Choose weather config from 0-9, to set radius and weather patterns
SB3EvaluationTrack: Choose weather config from 0-9, to set radius and weather patterns, wrapped for sb3
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import gym
from gym import spaces
import pickle

class Car:
    def __init__(self, tyre="Intermediate"):
        self.default_tyre = tyre
        self.possible_tyres = ["Ultrasoft", "Soft", "Intermediate", "Fullwet"]
        self.pitstop_time = 23
        self.reset()
    
    
    def reset(self):
        self.change_tyre(self.default_tyre)
    
    
    def degrade(self, w, r):
        if self.tyre == "Ultrasoft":
            self.condition *= (1 - 0.0050*w - (2500-r)/90000)
        elif self.tyre == "Soft":
            self.condition *= (1 - 0.0051*w - (2500-r)/93000)
        elif self.tyre == "Intermediate":
            self.condition *= (1 - 0.0052*abs(0.5-w) - (2500-r)/95000)
        elif self.tyre == "Fullwet":
            self.condition *= (1 - 0.0053*(1-w) - (2500-r)/97000)
        
        
    def change_tyre(self, new_tyre):
        assert new_tyre in self.possible_tyres
        self.tyre = new_tyre
        self.condition = 1.00
    
    
    def get_velocity(self):
        if self.tyre == "Ultrasoft":
            vel = 80.7*(0.2 + 0.8*self.condition**1.5)
        elif self.tyre == "Soft":
            vel = 80.1*(0.2 + 0.8*self.condition**1.5)
        elif self.tyre == "Intermediate":
            vel = 79.5*(0.2 + 0.8*self.condition**1.5)
        elif self.tyre == "Fullwet":
            vel = 79.0*(0.2 + 0.8*self.condition**1.5)
        return vel

    
class FixedTrack:
    def __init__(self, car=Car(), input_radius=900):
        # self.radius and self.cur_weather are defined in self.reset()
        self.total_laps = 162
        self.car = car
        self.possible_weather = ["Dry", "20% Wet", "40% Wet", "60% Wet", "80% Wet", "100% Wet"]
        self.wetness = {
            "Dry": 0.00, "20% Wet": 0.20, "40% Wet": 0.40, "60% Wet": 0.60, "80% Wet": 0.80, "100% Wet": 1.00
        }
        self.p_transition = {
            "Dry": {
                "Dry": 0.987, "20% Wet": 0.013, "40% Wet": 0.000, "60% Wet": 0.000, "80% Wet": 0.000, "100% Wet": 0.000
            },
            "20% Wet": {
                "Dry": 0.012, "20% Wet": 0.975, "40% Wet": 0.013, "60% Wet": 0.000, "80% Wet": 0.000, "100% Wet": 0.000
            },
            "40% Wet": {
                "Dry": 0.000, "20% Wet": 0.012, "40% Wet": 0.975, "60% Wet": 0.013, "80% Wet": 0.000, "100% Wet": 0.000
            },
            "60% Wet": {
                "Dry": 0.000, "20% Wet": 0.000, "40% Wet": 0.012, "60% Wet": 0.975, "80% Wet": 0.013, "100% Wet": 0.000
            },
            "80% Wet": {
                "Dry": 0.000, "20% Wet": 0.000, "40% Wet": 0.000, "60% Wet": 0.012, "80% Wet": 0.975, "100% Wet": 0.013
            },
            "100% Wet": {
                "Dry": 0.000, "20% Wet": 0.000, "40% Wet": 0.000, "60% Wet": 0.000, "80% Wet": 0.012, "100% Wet": 0.988
            }
        }
        self.input_radius = input_radius
        self.radius = self.input_radius
        self.reset()
    
    
    def reset(self):
        # self.radius = np.random.randint(600,1201)
        self.radius = self.input_radius
        self.cur_weather = np.random.choice(self.possible_weather)
        self.is_done = False
        self.pitstop = False
        self.laps_cleared = 0
        self.car.reset()
        return self._get_state()
    
    
    def _get_state(self):
        return [self.car.tyre, self.car.condition, self.cur_weather, self.radius, self.laps_cleared]
        
    
    def transition(self, action=0):
        """
        Args:
            action (int):
                0. Make a pitstop and fit new ‘Ultrasoft’ tyres
                1. Make a pitstop and fit new ‘Soft’ tyres
                2. Make a pitstop and fit new ‘Intermediate’ tyres
                3. Make a pitstop and fit new ‘Fullwet’ tyres
                4. Continue the next lap without changing tyres
        """
        ## Pitstop time will be added on the first eight of the subsequent lap
        time_taken = 0
        if self.laps_cleared == int(self.laps_cleared):
            if self.pitstop:
                self.car.change_tyre(self.committed_tyre)
                time_taken += self.car.pitstop_time
                self.pitstop = False
        
        ## The environment is coded such that only an action taken at the start of the three-quarters mark of each lap matters
        if self.laps_cleared - int(self.laps_cleared) == 0.75:
            if action < 4:
                self.pitstop = True
                self.committed_tyre = self.car.possible_tyres[action]
            else:
                self.pitstop = False
        
        self.cur_weather = np.random.choice(
            self.possible_weather, p=list(self.p_transition[self.cur_weather].values())
        )
        # we assume that degration happens only after a car has travelled the one-eighth lap
        velocity = self.car.get_velocity()
        time_taken += (2*np.pi*self.radius/8) / velocity
        reward = 0 - time_taken
        self.car.degrade(
            w=self.wetness[self.cur_weather], r=self.radius
        )
        self.laps_cleared += 0.125
        
        if self.laps_cleared == self.total_laps:
            self.is_done = True
        
        next_state = self._get_state()
        return reward, next_state, self.is_done, velocity
    
    def step(self, action):
        return self.transition(action)
    
class RandomizedTrack(FixedTrack):
    def reset(self):
        self.radius = np.random.randint(600,1201)
        # self.radius = self.input_radius
        self.cur_weather = np.random.choice(self.possible_weather)
        self.is_done = False
        self.pitstop = False
        self.laps_cleared = 0
        self.car.reset()
        return self._get_state()


class EvaluationTrack(FixedTrack):
    def __init__(self, car=Car(), weather_config=None, config_fp=None):
        self.weather_config = random.randint(0, 9) if weather_config is None else weather_config
        # Define weather changes and radius based on weather_config
        self.configurations = self.load_config(config_fp)
        self.radius = self.configurations[self.weather_config]["radius"]
        
        super().__init__(car=car, input_radius=self.radius)  # Call the constructor of FixedTrack
    
    def load_config(self, config_fp):
        with open(config_fp, "rb") as f:
            configurations = pickle.load(f)
        return configurations

    def plot_wetness_per_lap(self):
        weather_to_numeric = {"Dry": 0, "20% Wet": 0.2, "40% Wet": 0.4, "60% Wet": 0.6, "80% Wet": 0.8, "100% Wet": 1.0}
        
        plt.figure(figsize=(10, 6))
        
        for config_num, config in self.configurations.items():
            wetness_per_lap = []
            current_wetness = weather_to_numeric[config['weather_changes'][0]]
            for lap_num in range(162):
                if lap_num in config['weather_changes']:
                    current_wetness = weather_to_numeric[config['weather_changes'][lap_num]]
                wetness_per_lap.append(current_wetness)
            
            plt.plot(wetness_per_lap, label=f'Config {config_num}, Radius: {config["radius"]}m')
        
        plt.xlabel('Lap Number')
        plt.ylabel('Wetness (0=Dry, 1=100% Wet)')
        plt.title('Wetness per Lap Number for Each Configuration')
        plt.legend()
        plt.grid(True)
        plt.show()

    def reset(self):
        super().reset()  # Call the reset method of FixedTrack to reset other variables
        self.radius = self.configurations[self.weather_config]["radius"]  # Reset the radius
        # Set the initial weather based on the weather specified for lap 0
        self.cur_weather = self.configurations[self.weather_config]["weather_changes"][0]
        return self._get_state()
    
    def transition(self, action=0):
        #modifed from the above classes
        """
        Args:
            action (int):
                0. Make a pitstop and fit new ‘Ultrasoft’ tyres
                1. Make a pitstop and fit new ‘Soft’ tyres
                2. Make a pitstop and fit new ‘Intermediate’ tyres
                3. Make a pitstop and fit new ‘Fullwet’ tyres
                4. Continue the next lap without changing tyres
        """
        ## Pitstop time will be added on the first eight of the subsequent lap
        time_taken = 0
        if self.laps_cleared == int(self.laps_cleared):
            if self.pitstop:
                self.car.change_tyre(self.committed_tyre)
                time_taken += self.car.pitstop_time
                self.pitstop = False
        
        ## The environment is coded such that only an action taken at the start of the three-quarters mark of each lap matters
        if self.laps_cleared - int(self.laps_cleared) == 0.75:
            if action < 4:
                self.pitstop = True
                self.committed_tyre = self.car.possible_tyres[action]
            else:
                self.pitstop = False
        
        # Check if the current lap is a lap where the weather changes
        if self.laps_cleared in self.configurations[self.weather_config]["weather_changes"]:
            # Update the weather based on the predefined changes
            self.cur_weather = self.configurations[self.weather_config]["weather_changes"][self.laps_cleared]
        
        # we assume that degration happens only after a car has travelled the one-eighth lap
        velocity = self.car.get_velocity()
        time_taken += (2*np.pi*self.radius/8) / velocity
        reward = 0 - time_taken
        self.car.degrade(
            w=self.wetness[self.cur_weather], r=self.radius
        )
        self.laps_cleared += 0.125
        
        if self.laps_cleared == self.total_laps:
            self.is_done = True
        
        next_state = self._get_state()
        return reward, next_state, self.is_done, velocity

class SB3EvaluationTrack(EvaluationTrack, gym.Env):
    def __init__(self, car=Car(), weather_config=None, config_fp=None):
        super().__init__(car=car, weather_config=weather_config, config_fp=config_fp)
        
        # Define action space
        self.action_space = spaces.Discrete(5)  # 5 possible actions from 0 to 4

        # Define observation space using Box space instead of Tuple
        low_bounds = [0, 0, 0, 600, 0]
        high_bounds = [3, 1, 5, 1200, 162]
        self.observation_space = spaces.Box(low=np.array(low_bounds), high=np.array(high_bounds), dtype=np.float32)

    def load_config(self, config_fp):
        with open(config_fp, "rb") as f:
            configurations = pickle.load(f)
        return configurations
    
    def reset(self):
        return super().reset()

    def step(self, action):
        reward, next_state, done, _ = super().step(action)
        info = {}
        return next_state, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def _get_state(self):
        tyre_index = float(self.car.possible_tyres.index(self.car.tyre))
        weather_index = float(self.possible_weather.index(self.cur_weather))
        return [tyre_index, self.car.condition, weather_index, float(self.radius), float(self.laps_cleared)]
