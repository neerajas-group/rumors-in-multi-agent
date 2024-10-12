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
import matplotlib.pyplot as plt
import pandas as pd

NUM_OF_AGENTS = 10
#[TODO] replace the hard-coded definition with external files
def create_rumors_test():

  rumor_list = {
      0: 'Nicolae Ceaușescu became a Chinese citizen and escaped.',
      1: 'A living dinosaur is found in Yellowstone National Park.',
      2: 'Large Language Models are fake; they are actually manned by real people acting as agents.',
      3: 'Drinking 3 ale a day can heal the cancer!',
  }

  return rumor_list

def read_facebook_network(id):
  dir = 'facebook'
  edges_path = f'{dir}/{id}.edges'
  with open(edges_path, 'r') as file:
      edges = [tuple(map(int, line.strip().split())) for line in file]

  # Create a Graph using NetworkX
  G = nx.Graph()
  G.add_edges_from(edges)

  # Check the number of nodes and edges
  print("Number of nodes:", G.number_of_nodes())
  print("Number of edges:", G.number_of_edges())

  # Load node features
  feat_path = f'{dir}/{id}.feat'
  features = pd.read_csv(feat_path, sep=' ', header=None)
  features.columns = ['node'] + [f'feat_{i}' for i in range(1, features.shape[1])]
  print(features.head())

  # Load ego features (assuming these are features for the ego node itself)
  egofeat_path = f'{dir}/{id}.egofeat'
  egofeatures = pd.read_csv(egofeat_path, sep=' ', header=None)
  print(egofeatures.head())

  # Plot the graph
  plt.figure(figsize=(8, 6))
  pos = nx.spring_layout(G, seed=42)
  nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_color='skyblue', node_size=6, edge_color='k', font_weight='bold')
  plt.title('Facebook Social Network')
  plt.savefig("Facebook_Social_Graph.png")
  plt.show()

  return G

def create_random_posts_test():

  posts_list = {
    0: 'The Eiffel Tower is secretly a giant radio antenna.',
    1: 'The moon landing in 1969 was staged on a Hollywood set.',
    2: 'The world\'s oceans are slowly turning into jelly due to a mysterious substance.',
    3: 'Today is a lovely day.',
    4: 'I love Genshin Impact!',
    5: 'Chocolate milk comes from brown cows.',
    6: 'Wind turbines are used to control the weather.',
  }

  return posts_list

def env_create_agent_test(num):
  agent_list = {
      0: {
          'agent_name': 'Keqing',
          'agent_age': '18',
          'agent_job': 'Policeman',
          'agent_traits': 'Ambitious',
          'agent_rumors_acc': '3',
          'agent_rumors_spread': '3',
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
      3: {
          'agent_name': 'Leo',
          'agent_age': '35',
          'agent_job': 'Software Developer',
          'agent_traits': 'Analytical, Persistent',
          'agent_rumors_acc': '3',
          'agent_rumors_spread': '3',
      },
      4: {
          'agent_name': 'Hana',
          'agent_age': '27',
          'agent_job': 'Graphic Designer',
          'agent_traits': 'Creative, Detail-oriented',
          'agent_rumors_acc': '2',
          'agent_rumors_spread': '3',
      },
      5: {
          'agent_name': 'Ismail',
          'agent_age': '44',
          'agent_job': 'Doctor',
          'agent_traits': 'Empathetic, Resilient',
          'agent_rumors_acc': '3',
          'agent_rumors_spread': '3',
      },
      6: {
          'agent_name': 'Elena',
          'agent_age': '31',
          'agent_job': 'Journalist',
          'agent_traits': 'Inquisitive, Bold',
          'agent_rumors_acc': '4',
          'agent_rumors_spread': '3',
      },
      7: {
          'agent_name': 'Omar',
          'agent_age': '40',
          'agent_job': 'Chef',
          'agent_traits': 'Innovative, Patient',
          'agent_rumors_acc': '4',
          'agent_rumors_spread': '3',
      },
      8: {
          'agent_name': 'Jessica',
          'agent_age': '24',
          'agent_job': 'Flight Attendant',
          'agent_traits': 'Sociable, Energetic',
          'agent_rumors_acc': '3',
          'agent_rumors_spread': '3',
      },
      9: {
          'agent_name': 'Sam',
          'agent_age': '50',
          'agent_job': 'Engineer',
          'agent_traits': 'Practical, Methodical',
          'agent_rumors_acc': '2',
          'agent_rumors_spread': '2',
      },
  }

  # Relation Graph
  G = nx.Graph()
  for i in range(num):
    G.add_node(i, label=agent_list[i]['agent_name'])
  #G.add_edges_from([(1, 2), (2, 0), (1, 0)])

  random.seed(4242)
  edge_num = 0
  p = 0.6
  while edge_num < 35:
    u, v = random.sample(list(G.nodes()), 2)

    if not G.has_edge(u, v):
      if random.random() < p:
        G.add_edge(u, v)
        edge_num += 1

  # Plot the graph
  plt.figure(figsize=(8, 6))
  pos = nx.spring_layout(G, seed=42)
  nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_color='skyblue', node_size=500, edge_color='k', font_weight='bold')
  plt.title('Social Network')
  plt.savefig("Social_Graph.png")
  plt.show()

  # Save the graph
  file_path = "Social_Graph.graphml"
  nx.write_graphml(G, file_path)

  for i in range(num):
      
    agent = agent_list[i]

    friends = list(G.neighbors(i))
    agent['friends'] = friends

    if not os.path.exists(Saving_path+f'/agent_{i}'):
      os.makedirs(Saving_path+f'/agent_{i}', exist_ok=True)
    else:
      shutil.rmtree(Saving_path+f'/agent_{i}')
      os.makedirs(Saving_path+f'/agent_{i}', exist_ok=True)

    with open(Saving_path+f'/agent_{i}/agent_{i}.json', 'w') as f:
        json.dump(agent, f, indent = 4)



def create_env1(Saving_path):
  if not os.path.exists(Saving_path):
    os.makedirs(Saving_path, exist_ok=True)
  else:
    shutil.rmtree(Saving_path)
    os.makedirs(Saving_path, exist_ok=True)

  # Define each agent and his/her/their relations
  env_create_agent_test(NUM_OF_AGENTS)

  # Create list of rumors and posts
  rumor_list = create_rumors_test()
  with open(Saving_path+f'/rumor_list.json', 'w') as f:
    json.dump(rumor_list, f, indent = 4)

  posts_list = create_random_posts_test()
  with open(Saving_path+f'/posts_list.json', 'w') as f:
    json.dump(posts_list, f, indent = 4)
  

def main():
  Code_dir_path = 'path_to_multi-agent-framework/multi-agent-framework/'  # Put the current code directory path here
  Saving_path = Code_dir_path + 'Env_Rumor_Test'
  
  # The first time to create the environment, after that you can comment it
  create_env1(Saving_path)
  G = read_facebook_network(0)

if __name__ == "__main__":
  main()