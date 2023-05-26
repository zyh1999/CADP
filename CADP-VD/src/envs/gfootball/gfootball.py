import numpy as np
import gfootball.env as football_env
from gfootball.env import observation_preprocessing
from ..multiagentenv import MultiAgentEnv
import gym
import torch as th

import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

class GoogleFootballEnv(MultiAgentEnv):

    def __init__(
            self,
            write_full_episode_dumps=False,
            write_goal_dumps=False,
            dump_freq=0,
            render=False,
            time_limit=150,
            time_step=0,
            map_name='academy_counterattack_easy',
            stacked=False,
            representation="simple115v2",
            rewards='scoring',
            logdir='football_dumps',
            write_video=False,
            number_of_right_players_agent_controls=0,
            seed=0,
    ):
        if map_name == 'academy_3_vs_1_with_keeper':
            self.obs_dim = 26
            self.n_agents = 3
            self.n_enemies = 2
        elif map_name == 'academy_counterattack_hard':
            self.obs_dim = 34
            self.n_agents = 4
            self.n_enemies = 3
        elif map_name == 'academy_counterattack_easy':
            self.obs_dim = 30
            self.n_agents = 4
            self.n_enemies = 2
        else:
            raise ValueError("Not Support Map")

        self.write_full_episode_dumps = write_full_episode_dumps
        self.write_goal_dumps = write_goal_dumps
        self.dump_freq = dump_freq
        self.render = render
        self.episode_limit = time_limit
        self.time_step = time_step
        self.env_name = map_name
        self.stacked = stacked
        self.representation = representation
        self.rewards = rewards
        self.logdir = logdir
        self.write_video = write_video
        self.number_of_right_players_agent_controls = number_of_right_players_agent_controls
        self.seed = seed

        self.env = football_env.create_environment(
            write_full_episode_dumps=self.write_full_episode_dumps,
            write_goal_dumps=self.write_goal_dumps,
            env_name=self.env_name,
            stacked=self.stacked,
            representation=self.representation,
            rewards=self.rewards,
            logdir=self.logdir,
            render=self.render,
            write_video=self.write_video,
            dump_frequency=self.dump_freq,
            number_of_left_players_agent_controls=self.n_agents,
            number_of_right_players_agent_controls=self.number_of_right_players_agent_controls,
            channel_dimensions=(observation_preprocessing.SMM_WIDTH, observation_preprocessing.SMM_HEIGHT))
        self.env.seed(self.seed)

        obs_space_low = self.env.observation_space.low[0][:self.obs_dim]
        obs_space_high = self.env.observation_space.high[0][:self.obs_dim]

        self.action_space = [gym.spaces.Discrete(
            self.env.action_space.nvec[1]) for _ in range(self.n_agents)]
        self.observation_space = [
            gym.spaces.Box(low=obs_space_low, high=obs_space_high, dtype=self.env.observation_space.dtype) for _ in range(self.n_agents)
        ]

        self.n_actions = self.action_space[0].n
        self.obs = None

    def check_if_done(self):
        cur_obs = self.env.unwrapped.observation()[0]
        ball_loc = cur_obs['ball']
        ours_loc = cur_obs['left_team'][-self.n_agents:]

        if ball_loc[0] < 0 or any(ours_loc[:, 0] < 0):
            """
            This is based on the CDS paper:
            'We make a small and reasonable change to the half-court offensive scenarios: our players will lose if
            they or the ball returns to our half-court.'
            """
            return True

        return False

    def step(self, _actions):
        """Returns reward, terminated, info."""
        if th.is_tensor(_actions):
            actions = _actions.cpu().numpy()
        else:
            actions = _actions
        self.time_step += 1
        obs, rewards, done, info = self.env.step(actions.tolist())
        info["battle_won"] = False

        self.obs = obs

        if self.time_step >= self.episode_limit:
            info["episode_limit"] = True
            done = True

        if self.env_name in ['academy_3_vs_1_with_keeper', 'academy_counterattack_hard', 'academy_counterattack_easy']:
            if self.check_if_done():
                done = True

        """
        This is based on the CDS paper:
        "Environmental reward only occurs at the end of the game. 
        They will get +100 if they win, else get -1."
        If done=False, the reward is -1, 
        If done=True and sum(rewards)<=0 the reward is 1.
        If done=True and sum(rewards)>0 the reward is 100.
        """
        if sum(rewards) <= 0:
            return -int(done), done, info

        info["battle_won"] = True
        return 100, done, info

    def get_simple_obs(self, index=-1):
        full_obs = self.env.unwrapped.observation()[0]
        simple_obs = []

        if self.env_name == 'academy_3_vs_1_with_keeper':
            if index == -1:
                # global state, absolute position
                simple_obs.append(full_obs['left_team'][-self.n_agents:].reshape(-1))
                simple_obs.append(full_obs['left_team_direction'][-self.n_agents:].reshape(-1))

                simple_obs.append(full_obs['right_team'].reshape(-1))
                simple_obs.append(full_obs['right_team_direction'].reshape(-1))

                simple_obs.append(full_obs['ball'])
                simple_obs.append(full_obs['ball_direction'])
            else:
                # local state, relative position
                ego_position = full_obs['left_team'][-self.n_agents + index].reshape(-1)
                simple_obs.append(ego_position)
                simple_obs.append(
                    (np.delete(full_obs['left_team'][-self.n_agents:], index, axis=0) - ego_position).reshape(-1)
                )

                simple_obs.append(full_obs['left_team_direction'][-self.n_agents + index].reshape(-1))
                simple_obs.append(
                    np.delete(full_obs['left_team_direction'][-self.n_agents:], index, axis=0).reshape(-1)
                )

                simple_obs.append((full_obs['right_team'] - ego_position).reshape(-1))
                simple_obs.append(full_obs['right_team_direction'].reshape(-1))

                simple_obs.append(full_obs['ball'][:2] - ego_position)
                simple_obs.append(full_obs['ball'][-1].reshape(-1))
                simple_obs.append(full_obs['ball_direction'])

        elif self.env_name == 'academy_counterattack_hard':
            if index == -1:
                # global state, absolute position
                simple_obs.append(full_obs['left_team'][-self.n_agents:].reshape(-1))
                simple_obs.append(full_obs['left_team_direction'][-self.n_agents:].reshape(-1))

                simple_obs.append(full_obs['right_team'][0])
                simple_obs.append(full_obs['right_team'][1])
                simple_obs.append(full_obs['right_team'][2])
                simple_obs.append(full_obs['right_team_direction'][0])
                simple_obs.append(full_obs['right_team_direction'][1])
                simple_obs.append(full_obs['right_team_direction'][2])

                simple_obs.append(full_obs['ball'])
                simple_obs.append(full_obs['ball_direction'])

            else:
                # local state, relative position
                ego_position = full_obs['left_team'][-self.n_agents + index].reshape(-1)
                simple_obs.append(ego_position)
                simple_obs.append(
                    (np.delete(full_obs['left_team'][-self.n_agents:], index, axis=0) - ego_position).reshape(-1)
                )

                simple_obs.append(full_obs['left_team_direction'][-self.n_agents + index].reshape(-1))
                simple_obs.append(
                    np.delete(full_obs['left_team_direction'][-self.n_agents:], index, axis=0).reshape(-1)
                )

                simple_obs.append(full_obs['right_team'][0] - ego_position)
                simple_obs.append(full_obs['right_team'][1] - ego_position)
                simple_obs.append(full_obs['right_team'][2] - ego_position)
                simple_obs.append(full_obs['right_team_direction'][0])
                simple_obs.append(full_obs['right_team_direction'][1])
                simple_obs.append(full_obs['right_team_direction'][2])

                simple_obs.append(full_obs['ball'][:2] - ego_position)
                simple_obs.append(full_obs['ball'][-1].reshape(-1))
                simple_obs.append(full_obs['ball_direction'])

        elif self.env_name == 'academy_counterattack_easy':
            if index == -1:
                # global state, absolute position
                simple_obs.append(full_obs['left_team'][-self.n_agents:].reshape(-1))
                simple_obs.append(full_obs['left_team_direction'][-self.n_agents:].reshape(-1))

                simple_obs.append(full_obs['right_team'][0])
                simple_obs.append(full_obs['right_team'][1])
                simple_obs.append(full_obs['right_team_direction'][0])
                simple_obs.append(full_obs['right_team_direction'][1])
                simple_obs.append(full_obs['ball'])
                simple_obs.append(full_obs['ball_direction'])

            else:
                # local state, relative position
                ego_position = full_obs['left_team'][-self.n_agents + index].reshape(-1)
                simple_obs.append(ego_position)
                simple_obs.append(
                    (np.delete(full_obs['left_team'][-self.n_agents:], index, axis=0) - ego_position).reshape(-1)
                )

                simple_obs.append(full_obs['left_team_direction'][-self.n_agents + index].reshape(-1))
                simple_obs.append(
                    np.delete(full_obs['left_team_direction'][-self.n_agents:], index, axis=0).reshape(-1)
                )

                simple_obs.append(full_obs['right_team'][0] - ego_position)
                simple_obs.append(full_obs['right_team'][1] - ego_position)
                simple_obs.append(full_obs['right_team_direction'][0])
                simple_obs.append(full_obs['right_team_direction'][1])

                simple_obs.append(full_obs['ball'][:2] - ego_position)
                simple_obs.append(full_obs['ball'][-1].reshape(-1))
                simple_obs.append(full_obs['ball_direction'])
                
        simple_obs = np.concatenate(simple_obs)
        return simple_obs

    def get_obs(self):
        """Returns all agent observations in a list."""
        obs = [self.get_simple_obs(i) for i in range(self.n_agents)]
        return obs

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        return self.get_simple_obs(agent_id)

    def get_obs_size(self):
        """Returns the size of the observation."""
        return self.obs_dim

    def get_state(self):
        """Returns the global state."""
        return self.get_simple_obs(-1)

    def get_state_size(self):
        """Returns the size of the global state."""
        return self.obs_dim

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return [[1 for _ in range(self.n_actions)] for agent_id in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        return self.get_avail_actions()[agent_id]

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.action_space[0].n

    def reset(self):
        """Returns initial observations and states."""
        self.time_step = 0
        self.env.reset()

        return self.get_obs(), self.get_state()

    def render(self):
        pass

    def close(self):
        self.env.close()

    def seed(self):
        pass

    def save_replay(self):
        """Save a replay."""
        pass

    def get_env_info(self):
        env_info = super().get_env_info()
        env_info["n_agents"] = self.n_agents
        env_info["n_enemies"] = self.n_enemies

        return env_info