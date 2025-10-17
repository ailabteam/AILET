# File: qkd_env.py
# Version 3.0 - Three-Scenario Environment (Static, Dynamic, Realistic)
# Final, carefully implemented version.

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
    """
    A Gymnasium environment for simulating satellite QKD scheduling.
    It supports three scenarios:
    1. 'static': Only geometric constraints and minor random noise.
    2. 'dynamic': Adds time-correlated cloud cover.
    3. 'realistic': Adds link setup/switching costs to the dynamic scenario.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, num_ogs=5, scenario="static"):
        super(SatelliteQKDEnv, self).__init__()

        # --- Basic Setup ---
        self.ts = load.timescale()
        self.satellite = EarthSatellite(LINE1, LINE2, 'ISS (ZARYA)', self.ts)
        self.num_ogs = num_ogs
        
        # --- Configure Scenario ---
        if scenario not in ["static", "dynamic", "realistic"]:
            raise ValueError("Scenario must be 'static', 'dynamic', or 'realistic'")
        self.scenario = scenario
        
        self.is_dynamic_weather = (scenario == "dynamic" or scenario == "realistic")
        self.has_switching_cost = (scenario == "realistic")
        
        # Initialize attributes for all scenarios to prevent AttributeErrors
        self.cloud_status = np.zeros(self.num_ogs, dtype=np.float32)
        if self.is_dynamic_weather:
            self.cloud_chance_per_step = 0.01
            self.cloud_duration_range = (15, 30)
        
        self.setup_time_remaining = 0
        if self.has_switching_cost:
            self.SWITCHING_COST_MINUTES = 2
        
        # --- Internal State ---
        self.last_action = -1 # -1 denotes no previous action (or reset)
        
        # --- Action and Observation Space ---
        self.action_space = spaces.Discrete(self.num_ogs + 1)
        
        base_obs_dim = 4 + 3 * self.num_ogs
        obs_dim = base_obs_dim
        if self.is_dynamic_weather:
            obs_dim += self.num_ogs # Add cloud status vector
        if self.has_switching_cost:
            obs_dim += 1 # Add remaining setup time scalar
        
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        
        self.ogs_locations = []
        self.current_time = None
        self.end_time = None
        self.np_random = None

    def _get_obs(self):
        # Part 1: Time and Satellite Position
        minute_of_day = self.current_time.utc.hour * 60 + self.current_time.utc.minute
        geocentric = self.satellite.at(self.current_time)
        pos = wgs84.geographic_position_of(geocentric)
        obs_parts = [
            (minute_of_day / 1440.0) * 2 - 1, pos.latitude.degrees / 90.0,
            pos.longitude.degrees / 180.0, (pos.elevation.km / 1000.0) * 2 - 1,
        ]

        # Part 2: OGS Connection Information
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
        
        # Part 3: Scenario-specific observations
        if self.is_dynamic_weather:
            normalized_cloud_status = self.cloud_status / self.cloud_duration_range[1]
            obs_parts.extend(normalized_cloud_status)
        
        if self.has_switching_cost:
            normalized_setup_time = self.setup_time_remaining / self.SWITCHING_COST_MINUTES
            obs_parts.append(normalized_setup_time)
            
        return np.array(obs_parts, dtype=np.float32)

    def _calculate_reward(self, action):
        # Check for setup time cost first
        if self.has_switching_cost and self.setup_time_remaining > 0:
            return 0.0

        # Check for idle action
        if action == self.num_ogs: return 0.0
        
        # Check for clouds in relevant scenarios
        if self.is_dynamic_weather and self.cloud_status[action] > 0: return -5.0

        # Check for geometric visibility
        ogs_loc = self.ogs_locations[action]
        difference = self.satellite - ogs_loc
        topocentric = difference.at(self.current_time)
        alt, _, dist = topocentric.altaz()
        if alt.degrees < MIN_ELEVATION_DEGREES: return -1.0
        
        # Physics-based Key Rate Calculation
        L0_dB = 50; R0_bits_per_sec = 1e5
        path_loss_dB = 20 * math.log10(dist.km / 1000.0)
        total_loss_dB = L0_dB + path_loss_dB
        transmittance = 10**(-total_loss_dB / 10)
        key_rate = R0_bits_per_sec * transmittance * (SIMULATION_STEP.total_seconds())
        
        # Apply noise only in the simplest 'static' scenario
        if self.scenario == "static":
            key_rate *= self.np_random.uniform(0.8, 1.0)
            
        return key_rate

    def step(self, action):
        # Handle switching cost logic
        if self.has_switching_cost:
            # A switch occurs if the new action is a connection to a DIFFERENT OGS
            is_switching = (action != self.last_action) and (action != self.num_ogs)
            if is_switching:
                self.setup_time_remaining = self.SWITCHING_COST_MINUTES
            
            # Count down the setup time
            if self.setup_time_remaining > 0:
                self.setup_time_remaining -= 1

        reward = self._calculate_reward(action)
        self.current_time += SIMULATION_STEP
        
        if self.is_dynamic_weather:
            self._update_weather()
        
        self.last_action = action
        
        terminated = self.current_time.tt >= self.end_time.tt
        truncated = False
        observation = self._get_obs()
        
        return observation, reward, terminated, truncated, {}
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset state variables
        self.ogs_locations = self._generate_random_ogs(self.num_ogs)
        start_year = self.np_random.integers(2022, 2024)
        start_month = self.np_random.integers(1, 13)
        start_day = self.np_random.integers(1, 29)
        self.start_time = self.ts.utc(start_year, start_month, start_day)
        self.current_time = self.start_time
        self.end_time = self.start_time + timedelta(days=1)
        
        self.cloud_status.fill(0)
        self.setup_time_remaining = 0
        self.last_action = -1

        observation = self._get_obs()
        return observation, {}

    def _update_weather(self):
        self.cloud_status[self.cloud_status > 0] -= 1
        for i in range(self.num_ogs):
            if self.cloud_status[i] == 0 and self.np_random.random() < self.cloud_chance_per_step:
                self.cloud_status[i] = self.np_random.integers(
                    self.cloud_duration_range[0], self.cloud_duration_range[1] + 1)

    def _generate_random_ogs(self, num_ogs):
        if self.np_random is None: raise RuntimeError("RNG not initialized. Call reset() first.")
        ogs = []
        for i in range(num_ogs):
            lat = self.np_random.uniform(-60, 60); lon = self.np_random.uniform(-180, 180)
            ogs.append(wgs84.latlon(lat, lon, elevation_m=0))
        return ogs

    def render(self):
        if self.current_time is not None:
            print(f"Time: {self.current_time.utc_strftime('%Y-%m-%d %H:%M:%S')}")
            if self.is_dynamic_weather: print(f"Cloud Status: {self.cloud_status.astype(int)}")
            if self.has_switching_cost: print(f"Setup Time Left: {self.setup_time_remaining}")
