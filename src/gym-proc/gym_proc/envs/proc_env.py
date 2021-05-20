import datetime
from datetime import date, timedelta

import gym
import numpy as np
from gym.utils import seeding
from gym.vector.utils import spaces


class ProcurementEnv(gym.Env):
    metadata = {
        'render.modes': ['human']
    }
    stock = {}
    orders = []
    products = []
    procurements = []
    start_date = None
    current_date = None
    end_date = None
    stock_policy = {}
    debug = False
    kill = False

    def __init__(self, orders, products, start_date, end_date, stock_policy=None, debug=True):

        self.orders = orders
        self.products = products
        self.start_date = start_date
        self.current_date = start_date
        self.end_date = end_date
        if stock_policy:
            self.stock_policy = stock_policy
        self.debug = debug
        # Environment OpenAI metadata
        self.reward_range = (-np.inf, 0)
        self.action_space = spaces.Box(low=0.0, high=self.get_max_product_order_amount(),
                                       shape=(len(products), self.get_max_product_range()),
                                       dtype=np.int)  # products x order amounts
        self.observation_space = spaces.Box(low=0.0, high=self.get_max_product_order_amount(),
                                            shape=((end_date - start_date).days, len(products)),
                                            # matrix of days, products, amount
                                            dtype=np.int)
        stock = {}
        for p in products:
            stock[p.get_id()] = p.get_initial_stock()
        self.stock = stock

        if debug:
            print("init finished")

    def get_max_product_range(self):
        out = 0
        for p in self.products:
            if p.get_max_order_range() > out:
                out = p.get_max_order_range()
        return out

    def get_max_product_order_amount(self):
        out = 0
        for p in self.products:
            if p.max_order_amount() > out:
                out = p.max_order_amount()
        return out

    def seed(self, seed=None):  # pragma: no cover
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    ## action
    def step(self, action):
        self.current_date, self.stock = self.state

        # Handle Orders (- Stock)
        for o in self.orders:
            if o.get_delivery_data() == self.current_date:
                items = o.get_items()
                for i in items:
                    self.stock[i] -= items[i]
                    if self.debug:
                        print("order done", o)
                    if self.stock[i] < 0:
                        self.kill = True
                        if self.debug:
                            print("killed by low stock", self.stock)
                            return (self.current_date, self.stock), -999999999, False, {}

        # Handle new Procurements (by Agent)
        for idx, p in enumerate(self.products):
            order_amount = action[idx][0]
            delivery_date = self.current_date + datetime.timedelta(days=p.get_deliver_days())
            items = {}
            items[p.get_id()] = order_amount
            proc = Procurement("proc-"+str(self.current_date). replace("-", ""), "SID", delivery_date, self.current_date, items)
            self.procurements.append(proc)
            if self.debug:
                print("created procurement", proc)

        # Handle Procurements (+ Stock)
        for p in self.procurements:
            if p.get_delivery_date() == self.current_date:
                items = p.get_items()
                for i in items:
                    if self.debug:
                        print("handle procurement", p)
                    self.stock[i] += items[i]

        reward = self.calculate_reward(self.stock)
        if self.debug:
            print("date", self.current_date)
            print("stock", self.stock)
            print("reward", reward)
            print("----")
        self.current_date = self.current_date + timedelta(days=1)

        self.state = (self.current_date, self.stock)
        return self.state, reward, False, {}

    def calculate_reward(self, stock):
        out = 0
        for key in stock:
            out += stock[key]
        return out * -1

    def reset(self):
        self.kill = False
        self.current_date = self.start_date
        self.state = (self.current_date, self.stock)

        return self.state

    def done(self):
        if self.kill:
            return False
        return self.current_date < self.end_date


class Product(object):
    _id = None
    _max_order_range = [0]
    _max_order_amount = 0
    _deliver_days = 0
    _inital_stock = 0

    def __init__(self, id, max_order_amount, deliver_days, inital_stock):
        self._id = id
        self._deliver_days = deliver_days
        self._max_order_amount = max_order_amount
        self._max_order_range = range(0, max_order_amount,1)
        self._inital_stock = inital_stock

    def max_order_amount(self):
        return self._max_order_amount

    def get_id(self):
        return self._id
    def get_initial_stock(self):
        return self._inital_stock
    def get_deliver_days(self):
        return self._deliver_days

    def get_max_order_range(self):
        return len(self._max_order_range)

    def __str__(self):
        return "ID: {0}, MaxOrderRange: {1}, DeliveryDays: {2}, MaxOrderAmount: {3}".format(self._id,
                                                                                            self._max_order_range,
                                                                                            self._deliver_days,
                                                                                            self._max_order_amount,
                                                                                            )


class Transaction(object):
    _id = None
    _created_date = None
    _items = {}

    def __init__(self, id, created_date, items):
        self._id = id
        self._items = items
        self._created_date = created_date

    def get_id(self):
        return self._id

    def get_created_date(self):
        return self._created_date

    def get_items(self):
        return self._items


class Procurement(Transaction):
    _supplier_id = None
    _delivery_date = None

    def __init__(self, id, supplier_id, delivery_data, created_date, items):
        super().__init__(id, created_date, items)
        self._supplier_id = supplier_id
        self._delivery_date = delivery_data

        self._created_date = date.today()

    def get_delivery_date(self):
        return self._delivery_date

    def get_supplier_id(self):
        return self._supplier_id

    def __str__(self):
        return "ID: {0}, SupplierID: {1}, DeliveryDate: {2}, Items: {3}, CreatedAt {4}".format(self._id,
                                                                                               self._supplier_id,
                                                                                               self._delivery_date,
                                                                                               self.get_items(),
                                                                                               self.get_created_date())


class Order(Transaction):
    _customer_id = None
    _delivery_date = None

    def __init__(self, id, customer_id, delivery_date, created_date, items):
        super().__init__(id, created_date, items)
        self._customer_id = customer_id
        self._delivery_date = delivery_date

    def get_get_customer_id(self):
        return self._customer_id

    def get_delivery_data(self):
        return self._delivery_date

    def __str__(self):
        return "ID: {0}, CustomerID: {1}, DeliveryDate: {2}, Items: {3}, CreatedAt {4}".format(self._id,
                                                                                               self._customer_id,
                                                                                               self._delivery_date,
                                                                                               self.get_items(),
                                                                                               self.get_created_date())
