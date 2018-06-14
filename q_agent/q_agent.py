import math

import numpy as np
from absl import flags
from pysc2 import maps
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.lib import features

# feature views
_BOT_FEATURES_VIEW_INDEX = features.SCREEN_FEATURES.player_relative.index
_SELECTED_BOT_FEATURES_VIEW_INDEX = features.SCREEN_FEATURES.selected.index

# actions
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_MOVE_RAND = 1700
_MOVE_MIDDLE = 1701

# constants
_BACKGROUND = 0
_AI_SELF = 1
_AI_ALLIES = 2
_AI_NEUTRAL = 3
_AI_HOSTILE = 4
_SELECT_ALL = [0]
_NOT_QUEUED = [0]

# q-learning
GAMMA = 0.9
EPSILON_START = 0.1
EPSILON_END = 0.025
possible_actions = [
    _NO_OP,
    _SELECT_ARMY,
    _SELECT_POINT,
    _MOVE_SCREEN,
    _MOVE_RAND,
    _MOVE_MIDDLE
]
MAX_EPISODES = 35
MAX_STEPS = 400
EPS_DECAY = MAX_EPISODES * MAX_STEPS

# environment
FLAGS = flags.FLAGS
FLAGS(['run_sc2'])
VISUALIZATION = False
SAVE_REPLAY = False


def get_alpha(steps_done):
    return EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * steps_done / (0.7 * EPS_DECAY))


def get_state(obs, beacon_position_x, beacon_position_y):
    ai_view = obs.observation['screen'][_BOT_FEATURES_VIEW_INDEX]
    return is_bot_selected(obs), is_bot_on_beacon(ai_view, beacon_position_x, beacon_position_y)


def is_bot_on_beacon(ai_view, beacon_position_x, beacon_position_y):
    bot_position_x_array, bot_position_y_array = (ai_view == _AI_SELF).nonzero()
    bot_position_mean_x, bot_position_mean_y = bot_position_x_array.mean(), bot_position_y_array.mean()
    marine_on_beacon = np.min(beacon_position_x) <= bot_position_mean_x <= np.max(beacon_position_x) and \
                       np.min(beacon_position_y) <= bot_position_mean_y <= np.max(beacon_position_y)
    return int(marine_on_beacon)


def is_bot_selected(obs):
    ai_selected = obs.observation['screen'][_SELECTED_BOT_FEATURES_VIEW_INDEX]
    return int((ai_selected == 1).any())


def get_beacon_location(obs):
    ai_view = obs.observation['screen'][_BOT_FEATURES_VIEW_INDEX]
    beacon_x, beacon_y = (ai_view == _AI_NEUTRAL).nonzero()
    return [beacon_x, beacon_y]


class QTable(object):
    def __init__(self):
        self.actions = possible_actions
        self.states_list = set()
        self.q_table = np.zeros((0, len(possible_actions)))

    def get_action_index(self, state, steps):
        if np.random.rand() < get_alpha(steps):
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

    def update_q_table(self, state, next_state, action_index, reward, steps):
        if state not in self.states_list:
            self.add_state(state)
        if next_state not in self.states_list:
            self.add_state(next_state)
        state_idx = list(self.states_list).index(state)
        q_next_state, q_state = self.extract_states(action_index, next_state, state_idx)
        self.q_table[state_idx, action_index] += get_alpha(steps) * ((reward + (GAMMA * q_next_state)) - q_state)

    def extract_states(self, action_index, next_state, state_idx):
        next_state_idx = list(self.states_list).index(next_state)
        q_state = self.q_table[state_idx, action_index]
        q_next_state = self.q_table[next_state_idx].max()
        return q_next_state, q_state


def select_bot():
    return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])


def move_to_beacon(beacon_pos):
    beacon_x, beacon_y = beacon_pos[0].mean(), beacon_pos[1].mean()
    return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [beacon_y, beacon_x]])


def move_into_map_middle():
    return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [32, 32]])


def move_to_random_position(beacon_pos):
    beacon_position_max_x, beacon_position_max_y = beacon_pos[0].max(), beacon_pos[1].max()
    random_x, random_y = np.random.randint(beacon_position_max_x, 64), np.random.randint(beacon_position_max_y, 64)
    return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [random_y, random_x]])


def deselect_bot(obs):
    ai_view = obs.observation['screen'][_BOT_FEATURES_VIEW_INDEX]
    background_x_array, background_y_array = (ai_view == _BACKGROUND).nonzero()
    random_point = np.random.randint(0, len(background_x_array))
    background_x, background_y = background_x_array[random_point], background_y_array[random_point]
    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, [background_y, background_x]])


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
        self.q_table = QTable()

    def step(self, obs):
        super(QAgent, self).step(obs)
        beacon_pos = get_beacon_location(obs)
        state = get_state(obs, beacon_pos[0], beacon_pos[1])
        action_index = self.q_table.get_action_index(state, self.steps)
        action = select_action(action_index, beacon_pos, obs, state)
        return state, action_index, action


def main():
    with sc2_env.SC2Env(agent_race=None, bot_race=None, difficulty=None, map_name=maps.get('MoveToBeacon'),
                        visualize=VISUALIZATION) as env, \
            open("results", "a") as results_file:
        agent = QAgent()
        for i in range(MAX_EPISODES):
            episode_reward = 0
            obs = env.reset()
            for j in range(MAX_STEPS):
                state, action_index, action = agent.step(obs[0])
                obs = env.step(actions=[action])
                beacon_position = get_beacon_location(obs[0])
                next_state = get_state(obs[0], beacon_position[0], beacon_position[1])
                reward = obs[0].reward
                episode_reward += reward
                agent.q_table.update_q_table(state, next_state, action_index, reward, agent.steps)
            results_file.write('{}\n'.format(episode_reward))
        if SAVE_REPLAY:
            env.save_replay(QAgent.__name__)


if __name__ == "__main__":
    main()
