import time
from collections import namedtuple
import traci
import torch
import yaml
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as style
import pandas as pd
import ollama
import json
import random

from environment.traffic_environment import TrafficEnvironment
from algorithms.DQN.epsilon_greedy import EpsilonGreedy

def timed(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[TIME] {func.__name__} executed in {end - start:.3f} seconds")
        return result
    return wrapper

class TestTraffic:
    def __init__(self):
        self.modes = ['simple', 'adaptive', 'TSC']
        self.data = namedtuple('Data',
                                ('queue_length', 'waiting_time', 'co2',
                                 'nox', 'halting_vehicles', 'travel_time', 'arrived_vehicles'))
        self.parameters()
        self.env = TrafficEnvironment()
        self.action_selection = EpsilonGreedy(self.config, self.env)
        self.prompt_config = self.load_prompt_config()
        self.expert_actions = self.load_expert_actions()
        self.llm_cache = {}

    def parameters(self) -> dict:
        with open('config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

    def load_prompt_config(self) -> dict:
        with open('../algorithms/DQN/prompt2.yaml', 'r') as file:
            return yaml.safe_load(file)

    def load_expert_actions(self):
        with open('../algorithms/DQN/actions2.json', 'r') as file:
            data = json.load(file)
        return data['actions']

    def run(self):
        simple_data = self.simple()
        self.env.reset()
        actuated_data = self.actuated()
        self.env.reset()
        delay_data = self.delay_based()
        marl_data = self.marl()
        llm_data = self.llm()
        self.print_results(simple_data, actuated_data, delay_data, marl_data, llm_data)
        self.plot(simple_data, actuated_data, delay_data, marl_data, llm_data)

    @timed
    def simple(self):
        print("Testing Simple...")
        data = []
        for signal in self.env.signals:
            traci.trafficlight.setProgram(signal, "static")
        for warmup in range(self.env.config["WARMUP_STEPS"]):
            traci.simulationStep()

        steps = self.env.config["max_step"]
        for step in range(steps):
            traci.simulationStep()
            data.append(self.log_values())
        return data

    @timed
    def actuated(self):
        print("Testing Actuated...")
        data = []
        for signal in self.env.signals:
            traci.trafficlight.setProgram(signal, "static")
        for warmup in range(self.env.config["WARMUP_STEPS"]):
            traci.simulationStep()
        for signal in self.env.signals:
            traci.trafficlight.setProgram(signal, "actuated")

        steps = self.env.config["max_step"]
        for step in range(steps):
            traci.simulationStep()
            data.append(self.log_values())
        return data

    @timed
    def delay_based(self):
        print("Testing DelayBased...")
        data = []
        for signal in self.env.signals:
            traci.trafficlight.setProgram(signal, "static")
        for warmup in range(self.env.config["WARMUP_STEPS"]):
            traci.simulationStep()
        for signal in self.env.signals:
            traci.trafficlight.setProgram(signal, "delay")

        steps = self.env.config["max_step"]
        for step in range(steps):
            traci.simulationStep()
            data.append(self.log_values())
        return data

    @timed
    def marl(self):
        print("Testing RL...")
        data = []
        PATH = self.config["PATH_TEST"]
        agent = torch.load(PATH, map_location=torch.device('cpu'))
        agent.eval()
        self.config["EPSILON"] = 0

        for signal in self.env.signals:
            traci.trafficlight.setProgram(signal, "static")

        for episode in range(self.config["EPISODES"]):
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.config["DEVICE"]).unsqueeze(0)
            done = False
            for warmup in range(self.env.config["WARMUP_STEPS"]):
                traci.simulationStep()
            while not done:
                states = []
                actions = []
                for signal in self.env.network.instance.traffic_light:
                    state = self.env.get_state(signal)
                    state = torch.tensor(state, dtype=torch.float32, device=self.config["DEVICE"]).unsqueeze(0)
                    states.append(state)
                    action = self.action_selection.epsilon_greedy_selection(agent, state)
                    actions.append(action)
                observation, reward, terminated, truncated, episode_data = self.env.step(actions)
                data.append(episode_data)
                if terminated or truncated:
                    done = True

            data_shape = int((np.shape(np.array(data).flatten())[0]) / 7)
            data = np.reshape(data,(data_shape,7))
            return data

    @timed
    def llm(self):
        print("Testing LLM...")
        self.config["EPSILON"] = 0
        data = []
        for episode in range(self.config["EPISODES"]):
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.config["DEVICE"]).unsqueeze(0)
            done = False
            for warmup in range(self.env.config["WARMUP_STEPS"]):
                traci.simulationStep()
            llm_prev_actions = []
            while not done:
                states = []
                for signal in self.env.network.instance.traffic_light:
                    start_loop = time.perf_counter()

                    state = self.env.get_state(signal)
                    state = torch.tensor(state, dtype=torch.float32, device=self.config["DEVICE"]).unsqueeze(0)
                    states.append(state)
                    action_space = self.env.action_space.n
                    action = self.prompt_llm(state.tolist(), action_space, llm_prev_actions)
                    llm_prev_actions.append(action)

                    end_loop = time.perf_counter()
                    print(f"[TIME] Per-intersection loop executed in {end_loop - start_loop:.3f} seconds")
                    print(action)

                step_start = time.perf_counter()
                observation, reward, terminated, truncated, episode_data = self.env.step(action)
                step_end = time.perf_counter()
                print(f"[TIME] env.step execution: {step_end - step_start:.3f} seconds")

                data.append(episode_data)
                if terminated or truncated:
                    done = True

            data_shape = int((np.shape(np.array(data).flatten())[0]) / 7)
            data = np.reshape(data, (data_shape, 7))
            return data

    # ... a többi függvény változatlan marad (print_results, plot, prompt_llm, stb.)

if __name__ == '__main__':
    test = TestTraffic()
    test.run()
