import random
import matplotlib.pyplot as plt
import numpy as np

weeks = 35
start_inv = 12
start_price = 4
price_limit = 4
inventory_limit = 12                        # 12 included aswell as backorders
order_range = [i for i in range(-4, 5)]
c_h = 0.3
c_b = 6
profile_nr = 5
supply_lead_time = 1

# Funtions returns a list with the demand of each week depending on the profile chosen
def demand_profile(profile_nr, num_of_weeks):
    if profile_nr == 1:
        return [4 for i in range(num_of_weeks)]
    elif profile_nr == 2:
        l1 = [4 for i in range(19)]
        l2 = [6 for i in range(num_of_weeks-19)]
        return l1 + l2
    elif profile_nr == 3:
        increase = random.randint(0, 3)
        l1 = [4 for i in range(19)]
        l2 = [4+increase for i in range(num_of_weeks-19)]
        return l1 + l2
    elif profile_nr == 4:
        moment = random.randint(0, 4)
        increase = random.randint(0, 3)
        l1 = [4 for i in range(17+moment)]
        l2 = [4+increase for i in range(num_of_weeks-17-moment)]
        return l1 + l2
    elif profile_nr == 5:
        outcomes = [2, 4, 6, 8]
        probabilities = [1/4] * 4
        l = [np.random.choice(outcomes, p=probabilities) for i in range(num_of_weeks)]
        return l

demand = demand_profile(profile_nr, weeks)

""" This class of player implements the q-learning strategy """
class QPlayer:
    def __init__(self):
        self.inventory = [start_inv] + [0 for i in range(weeks-1)]
        self.backorders = [0 for i in range(weeks)]
        self.supply_orders = [0 for i in range(weeks)]
        self.costs = [-c_h*start_inv + start_price*demand[0]]
        self.inventory_offset = [start_inv]
        self.sold_goods = []
        self.price_set = [start_price]

    # Function updates the current inventory (for incoming and outcoming supply)
    def inventory_update(self, week, amount):
        self.inventory[week] = amount

    # Function updates the backorder for next week
    def backorders_update(self, week, backorder):
        self.backorders[week] += backorder

    # Function that places an order (based on inventory and backorders)
    def order_supply(self, week, week_demand, action):
        if self.inventory_offset[week + 1] < -inventory_limit:
            try:
                self.supply_orders[week + supply_lead_time] += week_demand + max(order_range)
            except:
                pass
        
        if self.inventory_offset[week + 1] >= -inventory_limit:
            next_week_inventory = (self.inventory_offset[week + 1] 
                                   + max((0, week_demand + action[0])))
            try:
                if next_week_inventory <= inventory_limit:
                    self.supply_orders[week + 1] += max((0, week_demand + action[0]))
                else:
                    self.supply_orders[week + 1] += (action[0] 
                                                     - (self.inventory_offset[week + 1] 
                                                        + action[0] - inventory_limit))
            except:
                pass

    # Reward function calculates the inventory and holding costs up to the week+1 (since decision at week has influence on this amount)
    def cost(self, week, price):
        self.costs.append(-c_h*self.inventory[week+1]-c_b*self.backorders[week+1]+
                          price*self.sold_goods[week])
        self.price_set.append(price)

    # Function executes one round (week) for one agent in the BeerGame
    def play_round(self, round, units_req):
        # Process the order    
        order = units_req + self.backorders[round]
        if order <= self.inventory[round] + self.supply_orders[round]:
            self.sold_goods.append(units_req + self.backorders[round])
            self.inventory_update(round+1, self.inventory[round] + self.supply_orders[round] 
                                  - units_req - self.backorders[round])
        else:
            self.sold_goods.append(self.inventory[round] + self.supply_orders[round])
            self.backorders_update(round+1, units_req + self.backorders[round]
                                  - self.inventory[round] - self.supply_orders[round])

        # Calculate the inventory offset
        self.inventory_offset.append(self.inventory[round+1] - self.backorders[round+1])