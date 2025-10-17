# File: qkd_env.py
# Version 2.1 - Bugfix for pre-flight check failures.
# Fixes AttributeError in static mode and TypeError in render().

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from skyfield.api import load, EarthSatellite, wgs84
from datetime import timedelta
import math

# --- Constants --- (Giữ nguyên)
LINE1 = '1 25544U 98067A   23318.49065972  .00007725  00000+0  14574-3 0  9990'
LINE2 = '2 25544  51.6416 261.2343 0006753  23.3835 301.7885 15.49390234427429'
SIMULATION_STEP = timedelta(minutes=1)
MIN_ELEVATION_DEGREES = 20.0

class SatelliteQKDEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, num_ogs=5, dynamic_weather=False):
        super(SatelliteQKDEnv, self).__init__()

        self.ts = load.timescale()
        self.satellite = EarthSatellite(LINE1, LINE2, 'ISS (ZARYA)', self.ts)
        self.num_ogs = num_ogs
        self.dynamic_weather = dynamic_weather
        
        # --- FIX #1: Initialize cloud_status REGARDLESS of mode ---
        # This ensures the attribute always exists.
        self.cloud_status = np.zeros(self.num_ogs, dtype=np.float32)
        
        if self.dynamic_weather:
            self.cloud_chance_per_step = 0.01
            self.cloud_duration_range = (15, 30)

        self.action_space = spaces.Discrete(self.num_ogs + 1)

        # --- FIX #1: Define observation space size based on the mode ---
        base_obs_dim = 4 + 3 * self.num_ogs
        if self.dynamic_weather:
            # Dynamic env includes cloud status in the observation
            obs_dim = base_obs_dim + self.num_ogs
        else:
            # Static env does not
            obs_dim = base_obs_dim
        
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.ogs_locations = []
        self.current_time = None
        self.end_time = None
        self.np_random = None

    def _update_weather(self):
        # (Giữ nguyên, logic này đã đúng)
        if not self.dynamic_weather:
            return
        self.cloud_status[self.cloud_status > 0] -= 1
        for i in range(self.num_ogs):
            if self.cloud_status[i] == 0 and self.np_random.random() < self.cloud_chance_per_step:
                self.cloud_status[i] = self.np_random.integers(
                    self.cloud_duration_range[0], self.cloud_duration_range[1] + 1
                )

    def _get_obs(self):
        # (Logic tính toán vị trí, alt, az, dist giữ nguyên)
        minute_of_day = self.current_time.utc.hour * 60 + self.current_time.utc.minute
        geocentric = self.satellite.at(self.current_time)
        pos = wgs84.geographic_position_of(geocentric)
        obs_parts = [
            (minute_of_day / 1440.0) * 2 - 1, pos.latitude.degrees / 90.0,
            pos.longitude.degrees / 180.0, (pos.elevation.km / 1000.0) * 2 - 1,
        ]
        for ogs_loc in self.ogs_locations:
            difference = self.satellite - ogs_loc
            topocentric = difference.at(self.current_time)
            alt, az, dist = topocentric.altaz()
            if alt.degrees < MIN_ELEVATION_DEGREES:
                obs_parts.extend([-1.0, -1.0, -1.0])
            else:
                obs_parts.extend([
                    alt.degrees / 90.0, az.degrees / 360.0 * 2 - 1,
                    1 - (dist.km / 4000.0)
                ])
        
        # --- FIX #1: Only add cloud status if in dynamic mode ---
        if self.dynamic_weather:
            normalized_cloud_status = self.cloud_status / self.cloud_duration_range[1]
            obs_parts.extend(normalized_cloud_status)
        
        return np.array(obs_parts, dtype=np.float32)

    def _calculate_reward(self, action):
        # (Giữ nguyên, logic này đã đúng)
        if action == self.num_ogs: return 0.0
        if self.dynamic_weather and self.cloud_status[action] > 0:
            return -5.0
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
        if not self.dynamic_weather:
            key_rate *= self.np_random.uniform(0.8, 1.0)
        return key_rate

    def step(self, action):
        # (Giữ nguyên, logic này đã đúng)
        reward = self._calculate_reward(action)
        self.current_time += SIMULATION_STEP
        self._update_weather()
        terminated = self.current_time.tt >= self.end_time.tt
        truncated = False
        observation = self._get_obs()
        info = {}
        return observation, reward, terminated, truncated, info
        
    def reset(self, seed=None, options=None):
        # (Logic reset thời gian, OGS giữ nguyên)
        super().reset(seed=seed)
        self.ogs_locations = self._generate_random_ogs(self.num_ogs)
        start_year = self.np_random.integers(2022, 2024)
        start_month = self.np_random.integers(1, 13)
        start_day = self.np_random.integers(1, 29)
        self.start_time = self.ts.utc(start_year, start_month, start_day)
        self.current_time = self.start_time
        self.end_time = self.start_time + timedelta(days=1)
        
        # Reset weather status (an toàn vì self.cloud_status luôn tồn tại)
        self.cloud_status.fill(0)

        observation = self._get_obs()
        info = {}
        return observation, info

    def _generate_random_ogs(self, num_ogs):
        # (Giữ nguyên)
        if self.np_random is None: raise RuntimeError("RNG not initialized. Call reset() first.")
        ogs = []
        for i in range(num_ogs):
            lat = self.np_random.uniform(-60, 60); lon = self.np_random.uniform(-180, 180)
            ogs.append(wgs84.latlon(lat, lon, elevation_m=0))
        return ogs

    def render(self):
        # --- FIX #2: Use "is not None" for checking Time object ---
        if self.current_time is not None:
            print(f"Time: {self.current_time.utc_strftime('%Y-%m-%d %H:%M:%S')}")
            if self.dynamic_weather:
                print(f"Cloud Status: {self.cloud_status.astype(int)}")
