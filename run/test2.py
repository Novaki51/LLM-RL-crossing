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

    def prompt_llm(self, state, action_space, llm_prev_actions, retries=5, delay=2):
        """
        Queries the locally running Llama model via Ollama to select the best action,
        ensuring compliance with the output format and valid action range.
        """

        prompt_template = self.prompt_config["prompt_template"]

        previous_actions = self.expert_actions[:43]

        prompt = f"""
        {prompt_template["content"]}
        Intersection data:
        Current State: {state}. Optimal: [10.0, 10.0, 10.0, 10.0]. 
        An expert agent used this action sequence to get the best results: {previous_actions}.
        Your previous chosen actions: {llm_prev_actions}. The last vector of the array is the last action pair chosen. 
        If the last 4 vectors of the previous actions array are the same, choose different action.
        Waiting time: {[traci.lane.getWaitingTime(laneID="-E6_0"),
                        traci.lane.getWaitingTime(laneID="-E0_0"),
                        traci.lane.getWaitingTime(laneID="-E3_0"),
                        traci.lane.getWaitingTime(laneID="lane_1_1_0"),
                        traci.lane.getWaitingTime(laneID="-E5_0"),
                        traci.lane.getWaitingTime(laneID="lane_0_3_0"),
                        traci.lane.getWaitingTime(laneID="-E4_0"),
                        traci.lane.getWaitingTime(laneID="E2_0")]}. Optimal values: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0].
        CO2 emission: {[traci.lane.getCO2Emission(laneID="-E6_0"),
                        traci.lane.getCO2Emission(laneID="-E0_0"),
                        traci.lane.getCO2Emission(laneID="-E3_0"),
                        traci.lane.getCO2Emission(laneID="lane_1_1_0"),
                        traci.lane.getCO2Emission(laneID="-E5_0"),
                        traci.lane.getCO2Emission(laneID="lane_0_3_0"),
                        traci.lane.getCO2Emission(laneID="-E4_0"),
                        traci.lane.getCO2Emission(laneID="E2_0")]}. Optimal values: [5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0].
        NOx emission: {[traci.lane.getNOxEmission(laneID="-E6_0"),
                        traci.lane.getNOxEmission(laneID="-E0_0"),
                        traci.lane.getNOxEmission(laneID="-E3_0"),
                        traci.lane.getNOxEmission(laneID="lane_1_1_0"),
                        traci.lane.getNOxEmission(laneID="-E5_0"),
                        traci.lane.getNOxEmission(laneID="lane_0_3_0"),
                        traci.lane.getNOxEmission(laneID="-E4_0"),
                        traci.lane.getNOxEmission(laneID="E2_0")]}. Optimal values: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0].
        Number of Halting Vehicles: {[traci.lane.getLastStepHaltingNumber(laneID="-E6_0"),
                                      traci.lane.getLastStepHaltingNumber(laneID="-E0_0"),
                                      traci.lane.getLastStepHaltingNumber(laneID="-E3_0"),
                                      traci.lane.getLastStepHaltingNumber(laneID="lane_1_1_0"),
                                      traci.lane.getLastStepHaltingNumber(laneID="-E5_0"),
                                      traci.lane.getLastStepHaltingNumber(laneID="lane_0_3_0"),
                                      traci.lane.getLastStepHaltingNumber(laneID="-E4_0"),
                                      traci.lane.getLastStepHaltingNumber(laneID="E2_0")]}. Optimal values: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0].
        Travel Time: {[traci.lane.getTraveltime(laneID="-E6_0"),
                       traci.lane.getTraveltime(laneID="-E0_0"),
                       traci.lane.getTraveltime(laneID="-E3_0"),
                       traci.lane.getTraveltime(laneID="lane_1_1_0"),
                       traci.lane.getTraveltime(laneID="-E5_0"),
                       traci.lane.getTraveltime(laneID="lane_0_3_0"),
                       traci.lane.getTraveltime(laneID="-E4_0"),
                       traci.lane.getTraveltime(laneID="E2_0"), ]}. Optimal values: [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0].
        If the values differ too much from the optimal values, try to choose different action than previously. 
        Return a valid JSON object strictly in this json format:

        json
        {{
            "action": [int, int]
        }}

        The action in both intersections must be one of the allowed values: 0 or 1.
        Choose an action in both intersections which is best for the defined goals. 
        Return ONLY the json format WITHOUT additional text.
        """

        for attempt in range(retries):
            try:
                response = ollama.chat(
                    model=prompt_template["model"],
                    messages=[{"role": "user", "content": prompt}]
                )
                text = response.get("message", {}).get("content", "{}")
                response_json = json.loads(text)
                #print(response_json)
                action = response_json.get("action")

                if (
                        isinstance(action, list) and
                        len(action) == 2 and
                        all(isinstance(a, int) for a in action)
                ):
                    max_phase_index = max(action_space) if isinstance(action_space, list) else action_space - 1
                    return [min(max(a, 0), max_phase_index) for a in action]

                print(f"[WARNING] Invalid action format on attempt {attempt + 1}: {action}")
            except Exception as e:
                print(f"[ERROR] LLM query failed on attempt {attempt + 1}: {e}")
                time.sleep(delay)

        print("[FALLBACK] Using default action [0, 0] after all retries failed.")
        return [0, 0]

    def print_results(self, simple_data_in, actuated_data_in, delay_data_in, marl_data_in, llm_data_in):

        simple_data_in = np.array(simple_data_in)
        actuated_data_in = np.array(actuated_data_in)
        delay_data_in = np.array(delay_data_in)
        marl_data_in = np.array(marl_data_in)
        llm_data_in = np.array(llm_data_in)
        simple_data = []
        actuated_data = []
        delay_data = []
        marl_data = []
        llm_data = []
        for i in range(6):
            simple_data.append(simple_data_in[:, i])
            actuated_data.append(actuated_data_in[:, i])
            delay_data.append(delay_data_in[:, i])
            marl_data.append(marl_data_in[:, i])
            llm_data.append(llm_data_in[:, i])

        print("\n")
        print("\t \t \t Waiting time \t \t \t \t AVG speed \t \t  \t \t CO2 \t \t \t \t \t \t  "
              "NOx \t \t \t \t  \t \t  Halting Vehicles \t \t \t \t Travel Time")
        print(
            f"Static    : {np.mean(simple_data[0])} \t \t {np.mean(simple_data[1])} \t \t \t  {np.mean(simple_data[2])} "
            f"\t \t  {np.mean(simple_data[3])} \t \t {np.sum(simple_data[4])} \t \t \t \t {np.mean(simple_data[5])}")
        print(
            f"Actuated  : {np.mean(actuated_data[0])} \t \t {np.mean(actuated_data[1])} \t \t \t  {np.mean(actuated_data[2])} "
            f"\t \t  {np.mean(actuated_data[3])} \t \t {np.sum(actuated_data[4])} \t \t \t \t {np.mean(actuated_data[5])}")
        print(
            f"Delayed   : {np.mean(delay_data[0])} \t \t {np.mean(delay_data[1])} \t \t \t {np.mean(delay_data[2])} "
            f"\t \t {np.mean(delay_data[3])} \t \t {np.sum(delay_data[4])} \t \t \t \t {np.mean(delay_data[5])}")
        print(f"MARL      : {np.mean(marl_data[0])} \t \t {np.mean(marl_data[1])} \t \t \t {np.mean(marl_data[2])} "
              f"\t \t {np.mean(marl_data[3])} \t \t {np.sum(marl_data[4])} \t \t \t \t {np.mean(marl_data[5])}")
        print(f"LLM       : {np.mean(llm_data[0])} \t \t {np.mean(llm_data[1])} \t \t \t {np.mean(llm_data[2])} "
              f"\t \t {np.mean(llm_data[3])} \t \t {np.sum(llm_data[4])} \t \t \t \t {np.mean(llm_data[5])}")

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
                    #print(action)
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
                    #print(f"[TIME] Per-intersection loop executed in {end_loop - start_loop:.3f} seconds")
                    #print(action)

                step_start = time.perf_counter()
                observation, reward, terminated, truncated, episode_data = self.env.step(action)
                step_end = time.perf_counter()
                #print(f"[TIME] env.step execution: {step_end - step_start:.3f} seconds")

                data.append(episode_data)
                if terminated or truncated:
                    done = True

            data_shape = int((np.shape(np.array(data).flatten())[0]) / 7)
            data = np.reshape(data, (data_shape, 7))
            return data

    # ... a többi függvény változatlan marad (print_results, plot, prompt_llm, stb.)

    def plot(self, static, actuated, delayed, marl, llm):
        "This describes which data is relevant in a certain test"
        data = 0
        window_size = 400
        static = np.array([row[data] for row in static])
        actuated = np.array([row[data] for row in actuated])
        delayed = np.array([row[data] for row in delayed])
        marl = np.array([row[data] for row in marl])
        llm = np.array([row[data] for row in llm])
        x = np.arange(len(static))

        mpl.rcParams['axes.facecolor'] = '#EEF3F9'
        static = pd.DataFrame(static, columns=['data'])
        actuated = pd.DataFrame(actuated, columns=['data'])
        delayed = pd.DataFrame(delayed, columns=['data'])
        marl = pd.DataFrame(marl, columns=['data'])
        llm = pd.DataFrame(llm, columns=['data'])

        static['smoothed_data'] = static['data'].rolling(window=window_size).mean()
        actuated['smoothed_data'] = actuated['data'].rolling(window=window_size).mean()
        delayed['smoothed_data'] = delayed['data'].rolling(window=window_size).mean()
        marl['smoothed_data'] = marl['data'].rolling(window=window_size).mean()
        llm['smoothed_data'] = llm['data'].rolling(window=window_size).mean()

        plt.figure(figsize=[10, 7])  # a new figure window
        plt.plot(x, static["smoothed_data"], label='Static',color='#000099')
        plt.plot(x, actuated["smoothed_data"], label='Actuated',color='#0066CC')
        plt.plot(x, delayed["smoothed_data"], label='Delay Based',color='#009999')
        plt.plot(x, marl["smoothed_data"], label='MARL',color='#f90001')
        plt.plot(x, llm["smoothed_data"], label='LLM',color='#000000')
        plt.legend(fontsize='large')
        plt.ylabel("Waiting time [s]")
        plt.grid(True, linewidth=1, linestyle='-', color='#ead1dc')
        plt.show()

    def filter_data(self, *args):
        filtered_data = None
        return filtered_data

    def log_values(self):
        waiting_time = []
        speed = []
        co2 = []
        nox = []
        halting_vehicles = []
        travel_time = []

        for lane in self.env.network.instance.lanes:
            waiting_time.append(traci.lane.getWaitingTime(lane))
            speed.append(traci.lane.getLastStepMeanSpeed(lane))
            co2.append(traci.lane.getCO2Emission(lane))
            nox.append(traci.lane.getNOxEmission(lane))
            halting_vehicles.append(traci.lane.getLastStepHaltingNumber(lane))
            travel_time.append(traci.lane.getTraveltime(lane))

        avg_waiting_time = np.mean(waiting_time)
        avg_speed = np.mean(speed)
        avg_co2 = np.mean(co2)
        avg_nox = np.mean(nox)
        avg_halting_vehicles = np.mean(halting_vehicles)
        avg_travel_time = np.mean(travel_time)
        arrived_vehicles = traci.simulation.getArrivedNumber()

        return [avg_waiting_time, avg_speed, avg_co2, avg_nox, avg_halting_vehicles, avg_travel_time, arrived_vehicles]


if __name__ == '__main__':
    test = TestTraffic()
    test.run()
