import time as t
import datetime
import os, sys
import random
from datetime import time, date

import numpy as np
import pandas as pd

file_path = 'src/gym-proc/'
sys.path.append(os.path.dirname(file_path))
import gym
import gym_proc
from gym_proc.envs import Order, Procurement, Product

# Method to generate dummy input data
def generate_dummy_data(start_date, amount_days, amount_products, amount_orders):
    end_date = start_date + datetime.timedelta(days=amount_days)
    products = []
    for i in range(0, amount_products):
        # id, max_order_amount, deliver_days, initial_stock
        products.append(Product("prod" + str(i), random.randrange(10, 100), 0, random.randrange(10, 100)))
    orders = []
    for i in range(0, amount_orders):
        # id, customer_id, delivery_date, created_date, items
        d = start_date + datetime.timedelta(days=random.randrange(0, amount_days))
        orders.append(Order('ord'+str(i), 'C'+str(i), d, d, {'prod'+str(random.randrange(0, amount_products)): random.randrange(50, 100)}))
    return start_date, end_date, products, orders

# A simple Random Agent
class RandomAgent(object):
    def __init__(self, action_space):
        self.name = "Random Agent"
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()
def custom_reward_function(stock=None, action=None, current_date = None, products=None, orders=None, procurements=None):
    out = 0
    for key in stock:
        out += stock[key]
    return out * 1
# Example how to run the environment
def run_rl():
    # init some sample orders and products
    orders = [Order('ord01', 'C01', datetime.date(2021, 10, 6), datetime.date(2021, 10, 6), {'prod01': 50}),
              Order('ord01', 'C01', datetime.date(2021, 10, 8), datetime.date(2021, 10, 8), {'prod01': 50})]

    products = [Product("prod01", 100, 0, 10), Product("prod02", 70, 0, 20)]

    start_date = datetime.date(2021, 10, 1)
    end_date = datetime.date(2021, 11, 1)
    # or generate dummy data
    #         start_date, end_date, products, orders = generate_dummy_data(datetime.date(2021, 1, 1), 90, 5,10)
    env = gym.make("Procurement-v0", orders=orders, products=products,
                   start_date=start_date, end_date=end_date, reward_function=custom_reward_function)

    seed = 0
    env.seed(seed)

    # Setup the agent
    agent = RandomAgent(env.action_space)
    reward = 0
    done = False
    env.scenario_name = agent.name
    observation = env.reset()
    episodes = 1

    for j in range(episodes):
        while env.done():
            # Action/Feedback
            action = agent.act(observation, reward, done)
            observation, reward, _, _ = env.step(action)
        env.plot(j)

        env.reset()


if __name__ == "__main__":
    print("Start Procurement RL")
    startTime = t.time()

    run_rl()

    print('process done in {0} seconds'.format(t.time() - startTime))
