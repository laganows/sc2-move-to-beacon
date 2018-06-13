import math

import numpy as np
from absl import flags
from pysc2 import maps
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.lib import features

# feature views
_AI_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_AI_SELECTED = features.SCREEN_FEATURES.selected.index

# actions
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_MOVE_RAND = 1000
_MOVE_MIDDLE = 2000

# constants
_BACKGROUND = 0
_AI_SELF = 1
_AI_ALLIES = 2
_AI_NEUTRAL = 3
_AI_HOSTILE = 4
_SELECT_ALL = [0]
_NOT_QUEUED = [0]

# q-learning
EPS_START = 0.9
EPS_END = 0.025
EPS_DECAY = 10000

possible_actions = [
    _NO_OP,
    _SELECT_ARMY,
    _SELECT_POINT,
    _MOVE_SCREEN,
    _MOVE_RAND,
    _MOVE_MIDDLE
]


def get_eps_threshold(steps_done):
    return EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)


def get_state(obs):
    ai_view = obs.observation['screen'][_AI_RELATIVE]
    beaconxs, beaconys = (ai_view == _AI_NEUTRAL).nonzero()
    marinexs, marineys = (ai_view == _AI_SELF).nonzero()
    marinex, mariney = marinexs.mean(), marineys.mean()

    marine_on_beacon = np.min(beaconxs) <= marinex <= np.max(beaconxs) and np.min(beaconys) <= mariney <= np.max(beaconys)

    ai_selected = obs.observation['screen'][_AI_SELECTED]
    marine_selected = int((ai_selected == 1).any())

    return (marine_selected, int(marine_on_beacon)), [beaconxs, beaconys]


class QTable(object):
    def __init__(self, actions, lr=0.01, reward_decay=0.9):
        self.lr = lr
        self.actions = actions
        self.reward_decay = reward_decay
        self.states_list = set()
        self.q_table = np.zeros((0, len(possible_actions)))

    def get_action_index(self, state):
        if np.random.rand() < get_eps_threshold(steps):
            return np.random.randint(0, len(self.actions))
        else:
            if state not in self.states_list:
                self.add_state(state)
            idx = list(self.states_list).index(state)
            q_values = self.q_table[idx]
            return int(np.argmax(q_values))

    def add_state(self, state):
        self.q_table = np.vstack([self.q_table, np.zeros((1, len(possible_actions)))])
        self.states_list.add(state)

    def update_qtable(self, state, next_state, action, reward):
        if state not in self.states_list:
            self.add_state(state)
        if next_state not in self.states_list:
            self.add_state(next_state)
        # how much reward
        state_idx = list(self.states_list).index(state)
        next_state_idx = list(self.states_list).index(next_state)
        # calculate q labels
        q_state = self.q_table[state_idx, action]
        q_next_state = self.q_table[next_state_idx].max()
        q_targets = reward + (self.reward_decay * q_next_state)
        # calculate our loss
        loss = q_targets - q_state
        # update the q value for this state/action pair
        self.q_table[state_idx, action] += self.lr * loss
        return loss


def select_bot():
    return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])


def move_to_beacon(beacon_pos):
    beacon_x, beacon_y = beacon_pos[0].mean(), beacon_pos[1].mean()
    return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [beacon_y, beacon_x]])


def move_into_map_middle():
    return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [32, 32]])


def move_to_random_position(beacon_pos):
    beacon_x, beacon_y = beacon_pos[0].max(), beacon_pos[1].max()
    movex, movey = np.random.randint(beacon_x, 64), np.random.randint(beacon_y, 64)
    return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [movey, movex]])


def deselect_bot(obs):
    ai_view = obs.observation['screen'][_AI_RELATIVE]
    backgroundxs, backgroundys = (ai_view == _BACKGROUND).nonzero()
    point = np.random.randint(0, len(backgroundxs))
    backgroundx, backgroundy = backgroundxs[point], backgroundys[point]
    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, [backgroundy, backgroundx]])


def do_nothing():
    return actions.FunctionCall(_NO_OP, [])


def select_action(action_index, beacon_pos, obs, state):
    if possible_actions[action_index] == _NO_OP:
        return do_nothing()
    elif state[0] and possible_actions[action_index] == _MOVE_SCREEN:
        return move_to_beacon(beacon_pos)
    elif possible_actions[action_index] == _SELECT_ARMY:
        return select_bot()
    elif state[0] and possible_actions[action_index] == _SELECT_POINT:
        return deselect_bot(obs)
    elif state[0] and possible_actions[action_index] == _MOVE_RAND:
        return move_to_random_position(beacon_pos)
    elif state[0] and possible_actions[action_index] == _MOVE_MIDDLE:
        return move_into_map_middle()
    return do_nothing()


class QAgent(base_agent.BaseAgent):
    def __init__(self):
        super(QAgent, self).__init__()
        self.q_table = QTable(possible_actions)

    def step(self, obs):
        super(QAgent, self).step(obs)
        state, beacon_pos = get_state(obs)
        action_index = self.q_table.get_action_index(state)
        action = select_action(action_index, beacon_pos, obs, state)
        return state, action_index, action


FLAGS = flags.FLAGS
FLAGS(['run_sc2'])

viz = False
save_replay = False
MAX_EPISODES = 35
MAX_STEPS = 400
steps = 0

# create a map
beacon_map = maps.get('MoveToBeacon')

# create an envirnoment
with sc2_env.SC2Env(agent_race=None,
                    bot_race=None,
                    difficulty=None,
                    map_name=beacon_map,
                    visualize=viz) as env:
    agent = QAgent()
    for i in range(MAX_EPISODES):
        print 'Starting episode {}'.format(i)
        ep_reward = 0
        obs = env.reset()
        for j in range(MAX_STEPS):
            steps += 1
            state, action, func = agent.step(obs[0])
            obs = env.step(actions=[func])
            next_state, _ = get_state(obs[0])
            reward = obs[0].reward
            ep_reward += reward
            loss = agent.q_table.update_qtable(state, next_state, action, reward)
        print 'Episode Reward: {}, Explore threshold: {}, Q loss: {}'.format(ep_reward, get_eps_threshold(steps), loss)
    if save_replay:
        env.save_replay(QAgent.__name__)
