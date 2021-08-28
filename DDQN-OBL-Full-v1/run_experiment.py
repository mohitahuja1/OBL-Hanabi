from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from third_party.dopamine import checkpointer
from third_party.dopamine import iteration_statistics
import gin.tf
import rl_env
from hanabi_learning_environment import pyhanabi
import numpy as np
import mlp_agent
import tensorflow as tf
import torch
import copy

from belief_model import *

LENIENT_SCORE = False


class ObservationStacker(object):
  """Class for stacking agent observations."""

  def __init__(self, history_size, observation_size, num_players):
    """Initializer for observation stacker.

    Args:
      history_size: int, number of time steps to stack.
      observation_size: int, size of observation vector on one time step.
      num_players: int, number of players.
    """
    self._history_size = history_size
    self._observation_size = observation_size
    self._num_players = num_players
    self._obs_stacks = list()
    for _ in range(0, self._num_players):
      self._obs_stacks.append(np.zeros(self._observation_size *
                                       self._history_size))

  def add_observation(self, observation, current_player):
    """Adds observation for the current player.

    Args:
      observation: observation vector for current player.
      current_player: int, current player id.
    """
    self._obs_stacks[current_player] = np.roll(self._obs_stacks[current_player],
                                               -self._observation_size)
    self._obs_stacks[current_player][(self._history_size - 1) *
                                     self._observation_size:] = observation

  def get_observation_stack(self, current_player):
    """Returns the stacked observation for current player.

    Args:
      current_player: int, current player id.
    """

    return self._obs_stacks[current_player]

  def reset_stack(self):
    """Resets the observation stacks to all zero."""

    for i in range(0, self._num_players):
      self._obs_stacks[i].fill(0.0)

  @property
  def history_size(self):
    """Returns number of steps to stack."""
    return self._history_size

  def observation_size(self):
    """Returns the size of the observation vector after history stacking."""
    return self._observation_size * self._history_size


def load_gin_configs(gin_files, gin_bindings):
  """Loads gin configuration files.

  Args:
    gin_files: A list of paths to the gin configuration files for this
      experiment.
    gin_bindings: List of gin parameter bindings to override the values in the
      config files.
  """
  gin.parse_config_files_and_bindings(gin_files,
                                      bindings=gin_bindings,
                                      skip_unknown=False)


@gin.configurable
def create_environment(game_type='Hanabi-Full', num_players=2):
  """Creates the Hanabi environment.

  Args:
    game_type: Type of game to play. Currently the following are supported:
      Hanabi-Full: Regular game.
      Hanabi-Small: The small version of Hanabi, with 2 cards and 2 colours.
    num_players: Int, number of players to play this game.

  Returns:
    A Hanabi environment.
  """
  return rl_env.make(
      environment_name=game_type, num_players=num_players, pyhanabi_path=None)


@gin.configurable
def create_obs_stacker(environment, belief_level, history_size=4):
  """Creates an observation stacker.

  Args:
    environment: environment object.
    history_size: int, number of steps to stack.

  Returns:
    An observation stacker object.
  """

  if belief_level == -1:
    shape = environment.vectorized_observation_shape()[0]
  elif belief_level in (0, 1, 2):
    shape = environment.vectorized_observation_shape()[0] + 125 # debug

  return ObservationStacker(history_size,
                            shape,
                            environment.players)

@gin.configurable
def create_agent(environment, obs_stacker, agent_type='DQN'):
  """Creates the Hanabi agent.

  Args:
    environment: The environment.
    obs_stacker: Observation stacker object.
    agent_type: str, type of agent to construct.

  Returns:
    An agent for playing Hanabi.

  Raises:
    ValueError: if an unknown agent type is requested.
  """
  if agent_type == 'DQN':
    return dqn_agent.DQNAgent(observation_size=obs_stacker.observation_size(),
                              num_actions=environment.num_moves(),
                              num_players=environment.players)
  elif agent_type == 'Rainbow':
    return rainbow_agent.RainbowAgent(
        observation_size=obs_stacker.observation_size(),
        num_actions=environment.num_moves(),
        num_players=environment.players)
  elif agent_type == 'QLearn':
    return qlearn_agent.QLearnAgent(
        observation_size=obs_stacker.observation_size(),
        num_actions=environment.num_moves(),
        num_players=environment.players)
  elif agent_type == 'ADRQN':
    return adrqn_agent.AdrqnAgent(
        observation_size=obs_stacker.observation_size(),
        num_actions=environment.num_moves(),
        num_players=environment.players)
  elif agent_type == 'MLP':
    return mlp_agent.MLPAgent(
        observation_size=obs_stacker.observation_size(),
        num_actions=environment.num_moves(),
        num_players=environment.players)
  else:
    raise ValueError('Expected valid agent_type, got {}'.format(agent_type))

def format_legal_moves(legal_moves, action_dim):
  """Returns formatted legal moves.

  This function takes a list of actions and converts it into a fixed size vector
  of size action_dim. If an action is legal, its position is set to 0 and -Inf
  otherwise.
  Ex: legal_moves = [0, 1, 3], action_dim = 5
      returns [0, 0, -Inf, 0, -Inf]

  Args:
    legal_moves: list of legal actions.
    action_dim: int, number of actions.

  Returns:
    a vector of size action_dim.
  """
  new_legal_moves = np.full(action_dim, -float('inf'))
  if legal_moves:
    new_legal_moves[legal_moves] = 0
  return new_legal_moves


def parse_observations(observations, num_actions, obs_stacker, belief_level, fictitious):
  """Deconstructs the rich observation data into relevant components.

  Args:
    observations: dict, containing full observations.
    num_actions: int, The number of available actions.
    obs_stacker: Observation stacker object.

  Returns:
    current_player: int, Whose turn it is.
    legal_moves: `np.array` of floats, of length num_actions, whose elements
      are -inf for indices corresponding to illegal moves and 0, for those
      corresponding to legal moves.
    observation_vector: Vectorized observation for the current player.
  """
  current_player = observations['current_player']
  current_player_observation = (
      observations['player_observations'][current_player])

  legal_moves = current_player_observation['legal_moves_as_int']
  legal_moves = format_legal_moves(legal_moves, num_actions)

  if belief_level == -1:
    observation_vector = current_player_observation['vectorized']
  elif belief_level in (0, 1):
    if fictitious == 0:
      hand_probas = get_hand_probas(current_player_observation, belief_level)
    elif fictitious == 1:
      hand_probas = [0]*125
    observation_vector = np.append(hand_probas, current_player_observation['vectorized'])
  elif belief_level == 2:
    if fictitious == 0:
      team_mate = 1 - current_player
      own_hand = get_own_hand_vector(observations['player_observations'][team_mate]['observed_hands'][1])
    elif fictitious == 1:
      own_hand = [0]*125
    observation_vector = np.append(own_hand, current_player_observation['vectorized'])
  obs_stacker.add_observation(observation_vector, current_player)
  observation_vector = obs_stacker.get_observation_stack(current_player)

  return current_player, legal_moves, observation_vector
  
def create_environment_copy(environment):
  state_copy = environment.state.copy()
  config={
    "colors":
        5,
    "ranks":
        5,
    "players":
        state_copy.num_players(),
    "max_information_tokens":
        8,
    "max_life_tokens":
        3,
    "observation_type":
        pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
    }
  environment_copy = rl_env.HanabiEnv(config)
  observations = environment_copy.reset(state_copy)
  return environment_copy, observations
  
def run_fictitious_transition(environment_copy, action, obs_stacker_copy, belief_level, agent, observation_vector):
    
    o_t = copy.deepcopy(observation_vector)
    a_t = copy.deepcopy(action)
    
    observations, reward1, is_done, _ = environment_copy.step(int(action))
    current_player, legal_moves, observation_vector = (
        parse_observations(observations, environment_copy.num_moves(), obs_stacker_copy, belief_level, 1))
    action = agent.fictitious_step(legal_moves, observation_vector)

    if is_done is True:
        r_t1 = reward1
        o_t1 = np.array([0.]*len(observation_vector))
        d_t1 = copy.deepcopy(is_done)
        l_t1 = np.array([0]*len(legal_moves))
        agent.write_fictitious_tuple((o_t, a_t, r_t1, o_t1, d_t1, l_t1))
        return

    observations, reward2, is_done, _ = environment_copy.step(int(action))
    current_player, legal_moves, observation_vector = (
        parse_observations(observations, environment_copy.num_moves(), obs_stacker_copy, belief_level, 1))

    r_t1 = reward1 + reward2
    d_t1 = copy.deepcopy(is_done)
    if is_done is True:
        o_t1 = np.array([0.]*len(observation_vector))
        l_t1 = np.array([0]*len(legal_moves))
    else:
        o_t1 = copy.deepcopy(observation_vector)
        l_t1 = copy.deepcopy(legal_moves)
    agent.write_fictitious_tuple((o_t, a_t, r_t1, o_t1, d_t1, l_t1))
    return

def run_one_episode(agent, environment, obs_stacker, belief_level):
  """Runs the agent on a single game of Hanabi in self-play mode.

  Args:
    agent: Agent playing Hanabi.
    environment: The Hanabi environment.
    obs_stacker: Observation stacker object.

  Returns:
    step_number: int, number of actions in this episode.
    total_reward: float, undiscounted return for this episode.
  """
#   begin_start_time = time.time_ns() # time
  obs_stacker.reset_stack()
  observations = environment.reset()
  if agent.eval_mode:
    fictitious_eval = 1
  else:
    fictitious_eval = 0
  current_player, legal_moves, observation_vector = (
      parse_observations(observations, environment.num_moves(), obs_stacker, belief_level, fictitious_eval))
  action, loss = agent.step(legal_moves, observation_vector)
  
  is_done = False
  total_reward = 0
  step_number = 0
  total_loss = loss
#   begin_period = time.time_ns() - begin_start_time # time
#   print("begin_period: ", begin_period) # time
  if not agent.eval_mode:
    # fictitious_start_time = time.time_ns() # time
    environment_copy, observations_copy = create_environment_copy(environment)
    obs_stacker_copy = copy.deepcopy(obs_stacker)
    run_fictitious_transition(environment_copy, action, obs_stacker_copy, belief_level, agent, observation_vector)
    # fictitious_period = time.time_ns() - fictitious_start_time # time
    # print("fictitious_period: ", fictitious_period) # time
  while not is_done:
    # step_start_time = time.time_ns() # time
    observations, reward, is_done, _ = environment.step(int(action))
    modified_reward = max(reward, 0) if LENIENT_SCORE else reward
    total_reward += modified_reward

    step_number += 1
    if is_done:
      break
    if agent.eval_mode:
      fictitious_eval = 1
    else:
      fictitious_eval = 0
    current_player, legal_moves, observation_vector = (
        parse_observations(observations, environment.num_moves(), obs_stacker, belief_level, fictitious_eval))
    action, loss = agent.step(legal_moves, observation_vector)
    # step_period = time.time_ns() - step_start_time # time
    # print("step_period: ", step_period) # time
    if not agent.eval_mode:
    #   fictitious2_start_time = time.time_ns() # time
      environment_copy, observations_copy = create_environment_copy(environment)
      obs_stacker_copy = copy.deepcopy(obs_stacker)
      run_fictitious_transition(environment_copy, action, obs_stacker_copy, belief_level, agent, observation_vector)
    #   fictitious2_period = time.time_ns() - fictitious2_start_time # time
    #   print("fictitious2_period: ", fictitious2_period) # time
    total_loss += loss

  agent.end_episode()

  return step_number, total_reward, total_loss

def run_one_phase(agent, environment, obs_stacker, min_steps, statistics,
                  run_mode_str, belief_level):
  """Runs the agent/environment loop until a desired number of steps.

  Args:
    agent: Agent playing hanabi.
    environment: environment object.
    obs_stacker: Observation stacker object.
    min_steps: int, minimum number of steps to generate in this phase.
    statistics: `IterationStatistics` object which records the experimental
      results.
    run_mode_str: str, describes the run mode for this agent.

  Returns:
    The number of steps taken in this phase, the sum of returns, and the
      number of episodes performed.
  """
  step_count = 0
  num_episodes = 0
  sum_returns = 0.
  sum_loss = 0.

  while step_count < min_steps:
    episode_length, episode_return, episode_loss = run_one_episode(agent, environment,
                                                     obs_stacker, belief_level)
    statistics.append({
        '{}_episode_lengths'.format(run_mode_str): episode_length,
        '{}_episode_returns'.format(run_mode_str): episode_return
    })

    step_count += episode_length
    sum_returns += episode_return
    num_episodes += 1
    sum_loss += episode_loss

  return step_count, sum_returns, num_episodes, sum_loss


@gin.configurable
def run_one_iteration(agent, environment, obs_stacker,
                      iteration, training_steps,
                      belief_level,
                      evaluate_every_n=100,
                      num_evaluation_games=100):
  """Runs one iteration of agent/environment interaction.

  An iteration involves running several episodes until a certain number of
  steps are obtained.

  Args:
    agent: Agent playing hanabi.
    environment: The Hanabi environment.
    obs_stacker: Observation stacker object.
    iteration: int, current iteration number, used as a global_step.
    training_steps: int, the number of training steps to perform.
    evaluate_every_n: int, frequency of evaluation.
    num_evaluation_games: int, number of games per evaluation.

  Returns:
    A dict containing summary statistics for this iteration.
  """
  start_time = time.time()

  statistics = iteration_statistics.IterationStatistics()

  # First perform the training phase, during which the agent learns.
  agent.eval_mode = False
  number_steps, sum_returns, num_episodes, sum_loss = (
      run_one_phase(agent, environment, obs_stacker, training_steps, statistics,
                    'train', belief_level))
  time_delta = time.time() - start_time
  tf.logging.info('Average training steps per second: %.2f',
                   number_steps / time_delta)

  average_return = sum_returns / num_episodes
  average_loss = sum_loss / num_episodes
  tf.logging.info('Average per episode return: %.2f', average_return)
  statistics.append({'average_return': average_return})
  statistics.append({'average_loss': average_loss})
  tf.logging.info('Average per episode loss: %.2f', average_loss)

  # Also run an evaluation phase if desired.
  if evaluate_every_n is not None and iteration % evaluate_every_n == 0:
    episode_data = []
    agent.eval_mode = True
    # Collect episode data for all games.
    for _ in range(num_evaluation_games):
      episode_data.append(run_one_episode(agent, environment, obs_stacker, belief_level))
    eval_episode_length, eval_episode_return, eval_episode_loss = map(np.mean, zip(*episode_data))

    statistics.append({
        'eval_episode_lengths': eval_episode_length,
        'eval_episode_returns': eval_episode_return,
        'eval_episode_losses': eval_episode_loss
    })
    tf.logging.info('Average eval. episode length: %.2f  Return: %.2f Loss: %.2f',
                    eval_episode_length, eval_episode_return, eval_episode_loss)
  else:
    statistics.append({
        'eval_episode_lengths': -1,
        'eval_episode_returns': -1
    })

  return statistics.data_lists


def log_experiment(experiment_logger, iteration, statistics,
                   logging_file_prefix='log', log_every_n=1):
  """Records the results of the current iteration.

  Args:
    experiment_logger: A `Logger` object.
    iteration: int, iteration number.
    statistics: Object containing statistics to log.
    logging_file_prefix: str, prefix to use for the log files.
    log_every_n: int, specifies logging frequency.
  """
  if iteration % log_every_n == 0:
    experiment_logger['iter{:d}'.format(iteration)] = statistics
    experiment_logger.log_to_file(logging_file_prefix, iteration)


@gin.configurable
def run_experiment(agent,
                   environment,
                   obs_stacker,
                   belief_level,
                   experiment_logger,
                   checkpoint_dir,
                   start_iteration,
                   experiment_checkpointer,
                   num_iterations=5000,
                   training_steps=2000,
                   logging_file_prefix='log',
                   log_every_n=1,
                   checkpoint_every_n=1):
  """Runs a full experiment, spread over multiple iterations."""
  tf.logging.info('Beginning training...')
  if num_iterations <= start_iteration:
    tf.logging.warning('num_iterations (%d) < start_iteration(%d)',
                       num_iterations, start_iteration)
    return

  for iteration in range(start_iteration, num_iterations):
    start_time = time.time()
    statistics = run_one_iteration(agent, environment, obs_stacker, iteration,
                                   training_steps, belief_level)

    start_time = time.time()
    log_experiment(experiment_logger, iteration, statistics,
                   logging_file_prefix, log_every_n)

    start_time = time.time()
    if int(iteration) % 1 == 0:
      torch.save(agent.mlp_target.state_dict(), "{}/model_{}.pt".format(checkpoint_dir, iteration))
      torch.save(agent, "{}/agent_{}.pt".format(checkpoint_dir, iteration))
