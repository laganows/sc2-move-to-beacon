from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

# feature view
_BOT_FEATURES_VIEW_INDEX = features.SCREEN_FEATURES.player_relative.index

# actions
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_TO_POSITION_ON_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id

# constants
_AI_NEUTRAL = 3
_SELECT_ALL = [0]
_NOT_QUEUED = [0]


def get_beacon_location(ai_relative_view):
    return (ai_relative_view == _AI_NEUTRAL).nonzero()


def move_to_beacon(beacon_x, beacon_y):
    beacon_position = [beacon_y.mean(), beacon_x.mean()]
    return actions.FunctionCall(_MOVE_TO_POSITION_ON_SCREEN, [_NOT_QUEUED, beacon_position])


def do_nothing():
    return actions.FunctionCall(_NO_OP, [])


def select_bot():
    return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])


def select_bot_action(obs):
    bot_view = obs.observation['screen'][_BOT_FEATURES_VIEW_INDEX]
    beacon_x, beacon_y = get_beacon_location(bot_view)
    if not beacon_y.any():
        return do_nothing()
    return move_to_beacon(beacon_x, beacon_y)


class SimpleAgent(base_agent.BaseAgent):
    def step(self, obs):
        super(SimpleAgent, self).step(obs)
        if _MOVE_TO_POSITION_ON_SCREEN not in obs.observation['available_actions']:
            return select_bot()
        else:
            return select_bot_action(obs)
