# File: qkd_env.py
# Version 1.4 - Fixed TypeError in step() by comparing time objects via their .tt attribute.

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from skyfield.api import load, EarthSatellite, wgs84
from datetime import timedelta
import math

# --- Constants ---
LINE1 = '1 25544U 98067A   23318.49065972  .00007725  00000+0  14574-3 0  9990'
LINE2 = '2 25544  51.6416 261.2343 0006753  23.3835 301.7885 15.49390234427429'
SIMULATION_STEP = timedelta(minutes=1)
MIN_ELEVATION_DEGREES = 20.0

class SatelliteQKDEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, num_ogs=5):
        super(SatelliteQKDEnv, self).__init__()
        self.ts = load.timescale()
        self.satellite = EarthSatellite(LINE1, LINE2, 'ISS (ZARYA)', self.ts)
        self.num_ogs = num_ogs
        self.ogs_locations = []
        self.action_space = spaces.Discrete(self.num_ogs + 1)
        obs_dim = 4 + 3 * self.num_ogs
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.current_time = None
        self.end_time = None
        self.np_random = None

    def _generate_random_ogs(self, num_ogs):
        if self.np_random is None:
            raise RuntimeError("self.np_random is not initialized. Did you call super().reset(seed=seed)?")
        ogs = []
        for i in range(num_ogs):
            lat = self.np_random.uniform(-60, 60)
            lon = self.np_random.uniform(-180, 180)
            ogs.append(wgs84.latlon(lat, lon, elevation_m=0))
        return ogs

    def _get_obs(self):
        minute_of_day = self.current_time.utc.hour * 60 + self.current_time.utc.minute
        geocentric = self.satellite.at(self.current_time)
        pos = wgs84.geographic_position_of(geocentric)
        lat, lon, alt = pos.latitude, pos.longitude, pos.elevation
        obs_parts = [
            minute_of_day / 1440.0, lat.degrees / 90.0,
            lon.degrees / 180.0, alt.km / 1000.0,
        ]
        for ogs_loc in self.ogs_locations:
            difference = self.satellite - ogs_loc
            topocentric = difference.at(self.current_time)
            alt, az, dist = topocentric.altaz()
            if alt.degrees < MIN_ELEVATION_DEGREES:
                obs_parts.extend([0, 0, 0])
            else:
                obs_parts.extend([
                    alt.degrees / 90.0, az.degrees / 360.0, dist.km / 4000.0
                ])
        return np.array(obs_parts, dtype=np.float32)

    def _calculate_reward(self, action):
        if action == self.num_ogs: return 0.0
        ogs_loc = self.ogs_locations[action]
        difference = self.satellite - ogs_loc
        topocentric = difference.at(self.current_time)
        alt, _, dist = topocentric.altaz()
        if alt.degrees < MIN_ELEVATION_DEGREES: return -1.0
        L0_dB = 50; R0_bits_per_sec = 1e5
        path_loss_dB = 20 * math.log10(dist.km / 1000.0)
        total_loss_dB = L0_dB + path_loss_dB
        transmittance = 10**(-total_loss_dB / 10)
        key_rate = R0_bits_per_sec * transmittance * (SIMULATION_STEP.total_seconds())
        if self.np_random is not None:
             key_rate *= self.np_random.uniform(0.8, 1.0)
        return key_rate

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ogs_locations = self._generate_random_ogs(self.num_ogs)
        start_year = self.np_random.integers(2022, 2024)
        start_month = self.np_random.integers(1, 13)
        start_day = self.np_random.integers(1, 29)
        self.start_time = self.ts.utc(start_year, start_month, start_day)
        self.current_time = self.start_time
        self.end_time = self.start_time + timedelta(days=1)
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        reward = self._calculate_reward(action)
        self.current_time += SIMULATION_STEP
        
        # *** FIX HERE: Compare time objects using their .tt attribute ***
        terminated = self.current_time.tt >= self.end_time.tt
        
        truncated = False
        observation = self._get_obs()
        info = {}
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.current_time:
            print(f"Time: {self.current_time.utc_strftime('%Y-%m-%d %H:%M:%S')}")
