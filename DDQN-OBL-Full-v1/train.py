from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

from third_party.dopamine import logger

import run_experiment
import os
import torch

FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    'gin_files', [],
    'List of paths to gin configuration files (e.g.'
    '"configs/hanabi_mlp.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1").')

flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')

flags.DEFINE_string('checkpoint_dir', '',
                    'Directory where checkpoint files should be saved. If '
                    'empty, no checkpoints will be saved.')
flags.DEFINE_string('checkpoint_file_prefix', 'ckpt',
                    'Prefix to use for the checkpoint files.')
flags.DEFINE_string('logging_dir', '',
                    'Directory where experiment data will be saved. If empty '
                    'no checkpoints will be saved.')
flags.DEFINE_string('logging_file_prefix', 'log',
                    'Prefix to use for the log files.')

flags.DEFINE_integer('belief_level', -1, "Belief level; -1: vanilla agent (no belief) 0: my belief about my hand 1: hand sample drawn from my belief about my hand 2: Own hand as belief sample")

flags.DEFINE_string('agent_file', 'previous_run/agent_28aug.pt',
                    'Path to the agent file related to previous run.')


def launch_experiment():
  """Launches the experiment.

  Specifically:
  - Load the gin configs and bindings.
  - Initialize the Logger object.
  - Initialize the environment.
  - Initialize the observation stacker.
  - Initialize the agent.
  - Reload from the latest checkpoint, if available, and initialize the
    Checkpointer object.
  - Run the experiment.
  """
  if FLAGS.base_dir == None:
    raise ValueError('--base_dir is None: please provide a path for '
                     'logs and checkpoints.')

  run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
  experiment_logger = logger.Logger('{}/logs'.format(FLAGS.base_dir))

  environment = run_experiment.create_environment()
  obs_stacker = run_experiment.create_obs_stacker(environment, FLAGS.belief_level)
  if os.path.exists(FLAGS.agent_file):
    print(f"Found agent at: {FLAGS.agent_file}")
    agent = torch.load(FLAGS.agent_file)
  else:
    print(f"Could not find agent at: {FLAGS.agent_file}")
    agent = run_experiment.create_agent(environment, obs_stacker)

  checkpoint_dir = '{}/checkpoints'.format(FLAGS.base_dir)
#   start_iteration, experiment_checkpointer = (
#       run_experiment.initialize_checkpointing(agent,
#                                               experiment_logger,
#                                               checkpoint_dir,
#                                               FLAGS.checkpoint_file_prefix))
  print("belief_level: ", FLAGS.belief_level)
  run_experiment.run_experiment(agent, environment,
                                obs_stacker, FLAGS.belief_level,
                                experiment_logger, checkpoint_dir,
                                start_iteration = 0, experiment_checkpointer = None,
                                logging_file_prefix=FLAGS.logging_file_prefix)


def main(unused_argv):
  """This main function acts as a wrapper around a gin-configurable experiment.

  Args:
    unused_argv: Arguments (unused).
  """
  launch_experiment()

if __name__ == '__main__':
  app.run(main)
