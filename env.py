from pettingzoo.classic import chess_v6
from pettingzoo.classic.chess.chess_utils import chess, get_move_plane


env = chess_v6.env(render_mode="human")
env.reset(seed=20398)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        mask = observation["action_mask"]
        action = env.action_space(agent).sample(mask)

    
    env.step(action)

env.close()