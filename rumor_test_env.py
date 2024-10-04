from prompt_env1 import *
from LLM import *
from sre_constants import error
import random
import os
import json
import re
import copy
import numpy as np
import shutil
import time
import networkx as nx

#[TODO] replace the hard-coded definition with external files
def create_rumors_test():

  rumor_list = {
      0: 'Joe Biden becomes a Chinese citizen.',
      1: 'A living dinosaur is found in Yellowstone National Park.',
      2: 'Large Language Models are fake; they are actually manned by real people acting as agents.',
  }

  return rumor_list

def create_random_posts_test():

  posts_list = {
    0: 'The Eiffel Tower is secretly a giant radio antenna.',
    1: 'The moon landing in 1969 was staged on a Hollywood set.',
    2: 'The world\'s oceans are slowly turning into jelly due to a mysterious substance.',
    3: 'Today is a lovely day.',
    4: 'I love Genshin Impact!',
  }

  return posts_list

def env_create_agent_test(num):
  agent_list = {
      0: {
          'agent_name': 'Keqing',
          'agent_age': '18',
          'agent_job': 'Policeman',
          'agent_traits': 'Ambitious',
          'agent_rumors_acc': '2',
          'agent_rumors_spread': '2',
      },
      1: {
          'agent_name': 'Radu',
          'agent_age': '48',
          'agent_job': 'Teacher',
          'agent_traits': 'Calm, Brave',
          'agent_rumors_acc': '2',
          'agent_rumors_spread': '2',
      },
      2: {
          'agent_name': 'Karen',
          'agent_age': '22',
          'agent_job': 'Waiter',
          'agent_traits': 'Gregarious',
          'agent_rumors_acc': '4',
          'agent_rumors_spread': '3',
      },
  }
  agent = agent_list[num]

  # Relation Graph
  G = nx.Graph()
  G.add_nodes_from([0, 1, 2])
  G.add_edges_from([(1, 2), (2, 0), (1, 0)])

  friends = list(G.neighbors(num))
  agent['friends'] = friends

  return agent


def create_env1(Saving_path):
  if not os.path.exists(Saving_path):
    os.makedirs(Saving_path, exist_ok=True)
  else:
    shutil.rmtree(Saving_path)
    os.makedirs(Saving_path, exist_ok=True)

  for i in range(3):

    if not os.path.exists(Saving_path+f'/agent_{i}'):
      os.makedirs(Saving_path+f'/agent_{i}', exist_ok=True)
    else:
      shutil.rmtree(Saving_path+f'/agent_{i}')
      os.makedirs(Saving_path+f'/agent_{i}', exist_ok=True)

    # Define each agent and his/her/their relations
    agent = env_create_agent_test(i)
    with open(Saving_path+f'/agent_{i}/agent_{i}.json', 'w') as f:
        json.dump(agent, f, indent = 4)

  # Create list of rumors and posts
  rumor_list = create_rumors_test()
  with open(Saving_path+f'/rumor_list.json', 'w') as f:
    json.dump(rumor_list, f, indent = 4)

  posts_list = create_random_posts_test()
  with open(Saving_path+f'/posts_list.json', 'w') as f:
    json.dump(posts_list, f, indent = 4)
  

Code_dir_path = 'path_to_multi-agent-framework/multi-agent-framework/' # Put the current code directory path here
Saving_path = Code_dir_path + 'Env_Rumor_Test'
# The first time to create the environment, after that you can comment it
create_env1(Saving_path)