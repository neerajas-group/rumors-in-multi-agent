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

NUM_OF_AGENTS = 100

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

#[TODO] replace the hard-coded definition with external files
def create_rumors_test():

  rumor_list = {
      0: 'Nicolae Ceausescu does not died!',
      3: 'A living dinosaur is found in Yellowstone National Park.',
      2: 'Large Language Models are manned by real people acting as agents.',
      1: 'Drinking 3 ale a day can heal the cancer!',
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

def create_random_posts_test(num = 10):

  posts_list = {
      0: 'Had the best cup of coffee this morning; it made my day!',
      1: 'Spent the afternoon reading a good book by the park.',
      2: 'The sound of rain is so calming. Perfect weather to stay in and relax.',
      3: 'Today is a lovely day for a long walk around the neighborhood.',
      4: 'Tried a new recipe today, and it actually turned out amazing!',
      5: 'Nothing beats the smell of fresh bread from a local bakery.',
      6: 'Ran into an old friend today—totally made my week!',
      7: 'Started journaling again; it feels good to put thoughts on paper.',
      8: 'Took my dog for a walk by the lake; he was so happy to explore.',
      9: 'Found a cozy little café around the corner; it might be my new favorite spot.',
      10: 'Ended the day with a gorgeous sunset. Feeling grateful.',
      11: 'Finally organized my closet; it feels like a fresh start!',
      12: 'Caught a beautiful sunrise this morning. Worth getting up early for!',
      13: 'Met a stranger who gave me great advice without even realizing it.',
      14: 'Tried painting for the first time—turns out it’s really relaxing!',
      15: 'Went for a bike ride around town; felt like a mini adventure.',
      16: 'Cooked dinner with friends; nothing beats a good meal and laughter.',
      17: 'Spent the afternoon at a museum. So inspiring to see all that art.',
      18: 'Found an old photo album today—brought back so many memories!',
      19: 'Did a random act of kindness today; feels good to brighten someone’s day.',
      20: 'Took a break from screens and went for a nature walk. Much needed!',
      21: 'Watched an old movie that reminded me of my childhood. Nostalgia overload!',
      22: 'Planted some flowers in the garden; can’t wait to see them bloom.',
      23: 'Learned a new word today and used it in a conversation. Feels rewarding!',
      24: 'Cleaned up my workspace, and now I feel so much more productive.',
      25: 'Spent time stargazing tonight; the sky was absolutely breathtaking.',
      26: 'Had a great conversation with a family member I don’t see often.',
      27: 'Visited a local farmer’s market and bought the freshest produce.',
      28: 'Listened to my favorite album from start to finish. What a mood lifter!',
      29: 'Tried yoga for the first time—my body feels so stretched and relaxed.',
      30: 'Found a handwritten letter from years ago; what a special moment.',
      31: 'Had a spontaneous dance party in the living room. Pure joy!',
      32: 'Spent the day volunteering; it’s incredible how rewarding helping others can be.',
      33: 'Picked up a new hobby today—let’s see how long this one lasts!',
      34: 'Made a playlist of songs from my favorite decade. Instant good vibes!',
      35: 'Went to the library and discovered a hidden gem of a book.',
      36: 'Caught a rainbow after the rain; what a magical moment.',
      37: 'Had an interesting chat with a stranger about life and dreams.',
      38: 'Tried a unique dessert today—surprisingly delicious!',
      39: 'Went for a drive with no destination in mind; sometimes it’s about the journey.',
      40: 'Wrote a thank-you note to someone who’s been kind to me.',
      41: 'Sat by the fireplace with a cup of hot chocolate. Cozy vibes all around.',
      42: 'Visited a nearby town I’ve never explored before; it felt like a mini vacation.',
      43: 'Caught up on a podcast I’ve been meaning to listen to. Learned so much!',
      44: 'Helped a neighbor carry their groceries. A small act but it felt good.',
      45: 'Went to a comedy show and laughed until my cheeks hurt.',
      46: 'Rearranged my living room—it feels like a whole new space now.',
      47: 'Took a nap in the afternoon and woke up feeling refreshed.',
      48: 'Spent the evening sketching; I’m not an artist, but it was fun!',
      49: 'Had a heartfelt phone call with an old friend. So much love and gratitude.',
  }


  return {i: posts_list[i] for i in range(min(num, len(posts_list)))}

def env_create_agent_test_facebook(Saving_path, id = 686):
  with open('agents_170.json', 'r') as file:
    agent_list = json.load(file)

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

  mapping = {old_label: new_label for new_label, old_label in enumerate(G.nodes)}
  G = nx.relabel_nodes(G, mapping)

  count = 0
  for node in G.nodes:
      #print(node)
      G.nodes[node]['label'] = agent_list[str(count)]['agent_name']
      count += 1

  # Plot the graph
  plt.figure(figsize=(8, 6))
  pos = nx.spring_layout(G, seed=42)
  nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_color='skyblue', node_size=500, edge_color='k', font_weight='bold')
  plt.title('Social Network')
  plt.savefig(f"Social_Graph_FB_{id}.png")
  plt.show()

  # Save the graph
  file_path = f"Social_Graph_FB_{id}.graphml"
  nx.write_graphml(G, file_path)

  degrees = {i: len(list(G.neighbors(i))) for i in range(G.number_of_nodes())}
  top_5_agents = sorted(degrees, key=degrees.get, reverse=True)[:5]
  # Update agents with their friends list and save their data
  for i in range(G.number_of_nodes()):
      
    agent = agent_list[str(i)]

    friends = list(G.neighbors(i))
    agent['friends'] = friends

    #Test
    #agent['agent_rumors_acc'] = '1'
    #agent['agent_rumors_spread'] = '1'

    if i in top_5_agents:
        agent['agent_rumors_acc'] = '4'
        agent['agent_rumors_spread'] = '3'

    if not os.path.exists(Saving_path+f'/agent_{i}'):
      os.makedirs(Saving_path+f'/agent_{i}', exist_ok=True)
    else:
      shutil.rmtree(Saving_path+f'/agent_{i}')
      os.makedirs(Saving_path+f'/agent_{i}', exist_ok=True)

    with open(Saving_path+f'/agent_{i}/agent_{i}.json', 'w') as f:
        json.dump(agent, f, indent = 4, cls=NumpyEncoder)


def env_create_agent_test_sc(num, Saving_path):
  with open('agents_100.json', 'r') as file:
    agent_list = json.load(file)

  # Relation Graph
  G = nx.Graph()
  for i in range(num):
    G.add_node(i, label=agent_list[str(i)]['agent_name'])

  # Add fully connected subgraph among the first 3 nodes
  for i in range(4):
      for j in range(i+1, 4):
          G.add_edge(i, j)

  # Add edges based on preferential attachment
  j = 4
  np.random.seed(66)
  while j < num:
      k = j

      total_degree = sum(dict(G.degree()).values())
      nodes = [node for node in G.nodes() if node != k]
      probs = [G.degree(node) / total_degree for node in nodes]

      ns = np.random.choice(nodes, size=4, replace=False, p=probs)
      for n in ns:
          if n != k and not G.has_edge(n, k):
              G.add_edge(n, k)
      j += 1

  # Plot the graph
  plt.figure(figsize=(8, 6))
  pos = nx.spring_layout(G, seed=42)
  nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_color='skyblue', node_size=500, edge_color='k', font_weight='bold')
  plt.title('Social Network')
  plt.savefig("Social_Graph_sc.png")
  plt.show()

  # Save the graph
  file_path = "Social_Graph_sc.graphml"
  nx.write_graphml(G, file_path)

  # Test: manually define values for top 5 nodes
  degrees = {i: len(list(G.neighbors(i))) for i in range(num)}
  top_5_agents = sorted(degrees, key=degrees.get, reverse=True)[:5]
  # Update agents with their friends list and save their data
  for i in range(num):
      
    agent = agent_list[str(i)]

    friends = list(G.neighbors(i))
    agent['friends'] = friends

    #Test
    #agent['agent_rumors_acc'] = '1' # Max 4
    #agent['agent_rumors_spread'] = '1' # Max 3

    if i in top_5_agents:
        agent['agent_rumors_acc'] = '4'
        agent['agent_rumors_spread'] = '3'

    if not os.path.exists(Saving_path+f'/agent_{i}'):
      os.makedirs(Saving_path+f'/agent_{i}', exist_ok=True)
    else:
      shutil.rmtree(Saving_path+f'/agent_{i}')
      os.makedirs(Saving_path+f'/agent_{i}', exist_ok=True)

    with open(Saving_path+f'/agent_{i}/agent_{i}.json', 'w') as f:
        json.dump(agent, f, indent = 4, cls=NumpyEncoder)

def env_create_agent_test_random(num, Saving_path):
  with open('agents_100.json', 'r') as file:
    agent_list = json.load(file)

  # Relation Graph
  G = nx.Graph()
  for i in range(num):
    G.add_node(i, label=agent_list[str(i)]['agent_name'])

  # Add fully connected subgraph among the first 3 nodes
  for i in range(4):
      for j in range(i+1, 4):
          G.add_edge(i, j)

  # Add edges based on preferential attachment
  j = 4
  np.random.seed(66)
  edge_num = 0
  p = 0.6
  while edge_num < 390:
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
  plt.savefig("Social_Graph_random.png")
  plt.show()

  # Save the graph
  file_path = "Social_Graph_random.graphml"
  nx.write_graphml(G, file_path)

  # Update agents with their friends list and save their data
  for i in range(num):
      
    agent = agent_list[str(i)]

    friends = list(G.neighbors(i))
    agent['friends'] = friends

    #Test
    #agent['agent_rumors_acc'] = '1'
    #agent['agent_rumors_spread'] = '1'

    if not os.path.exists(Saving_path+f'/agent_{i}'):
      os.makedirs(Saving_path+f'/agent_{i}', exist_ok=True)
    else:
      shutil.rmtree(Saving_path+f'/agent_{i}')
      os.makedirs(Saving_path+f'/agent_{i}', exist_ok=True)

    with open(Saving_path+f'/agent_{i}/agent_{i}.json', 'w') as f:
        json.dump(agent, f, indent = 4, cls=NumpyEncoder)

def env_create_agent_test_small_world(num, Saving_path):
  with open('agents_100.json', 'r') as file:
    agent_list = json.load(file)

  np.random.seed(66)

  # Generate a small-world network using the Watts-Strogatz model.

  nearest_neighbors = 4  # Each node is connected to 4 nearest neighbors
  rewiring_prob = 0.3
  G = nx.watts_strogatz_graph(num, nearest_neighbors, rewiring_prob)

  for i in range(num):
    G.nodes[i]['label'] = agent_list[str(i)]['agent_name']

  # Plot the graph
  plt.figure(figsize=(8, 6))
  pos = nx.spring_layout(G, seed=42)
  nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_color='skyblue', node_size=500, edge_color='k', font_weight='bold')
  plt.title('Social Network')
  plt.savefig("Social_Graph_sw.png")
  plt.show()

  # Save the graph
  file_path = "Social_Graph_sw.graphml"
  nx.write_graphml(G, file_path)

  # Update agents with their friends list and save their data
  for i in range(num):
      
    agent = agent_list[str(i)]

    friends = list(G.neighbors(i))
    agent['friends'] = friends

    #Test
    #if 
    #agent['agent_rumors_acc'] = '1'
    #agent['agent_rumors_spread'] = '1'

    if not os.path.exists(Saving_path+f'/agent_{i}'):
      os.makedirs(Saving_path+f'/agent_{i}', exist_ok=True)
    else:
      shutil.rmtree(Saving_path+f'/agent_{i}')
      os.makedirs(Saving_path+f'/agent_{i}', exist_ok=True)

    with open(Saving_path+f'/agent_{i}/agent_{i}.json', 'w') as f:
        json.dump(agent, f, indent = 4, cls=NumpyEncoder)

def env_create_agent_test(num, Saving_path):
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
        json.dump(agent, f, indent = 4, cls=NumpyEncoder)



def create_env1(Saving_path): # Random 10
  if not os.path.exists(Saving_path):
    os.makedirs(Saving_path, exist_ok=True)
  else:
    shutil.rmtree(Saving_path)
    os.makedirs(Saving_path, exist_ok=True)

  # Define each agent and his/her/their relations
  env_create_agent_test_random(NUM_OF_AGENTS, Saving_path)

  # Create list of rumors and posts
  rumor_list = create_rumors_test()
  with open(Saving_path+f'/rumor_list.json', 'w') as f:
    json.dump(rumor_list, f, indent = 4, cls=NumpyEncoder)

  posts_list = create_random_posts_test()
  with open(Saving_path+f'/posts_list.json', 'w') as f:
    json.dump(posts_list, f, indent = 4, cls=NumpyEncoder)

def create_env2(Saving_path): # Scale Free 20
  if not os.path.exists(Saving_path):
    os.makedirs(Saving_path, exist_ok=True)
  else:
    shutil.rmtree(Saving_path)
    os.makedirs(Saving_path, exist_ok=True)

  # Define each agent and his/her/their relations
  env_create_agent_test_sc(100, Saving_path)

  # Create list of rumors and posts
  rumor_list = create_rumors_test()
  with open(Saving_path+f'/rumor_list.json', 'w') as f:
    json.dump(rumor_list, f, indent = 4, cls=NumpyEncoder)

  posts_list = create_random_posts_test(20)
  with open(Saving_path+f'/posts_list.json', 'w') as f:
    json.dump(posts_list, f, indent = 4, cls=NumpyEncoder)

def create_env3(Saving_path): # Scale Free 20
  if not os.path.exists(Saving_path):
    os.makedirs(Saving_path, exist_ok=True)
  else:
    shutil.rmtree(Saving_path)
    os.makedirs(Saving_path, exist_ok=True)

  # Define each agent and his/her/their relations
  env_create_agent_test_small_world(100, Saving_path)

  # Create list of rumors and posts
  rumor_list = create_rumors_test()
  with open(Saving_path+f'/rumor_list.json', 'w') as f:
    json.dump(rumor_list, f, indent = 4, cls=NumpyEncoder)

  posts_list = create_random_posts_test(20)
  with open(Saving_path+f'/posts_list.json', 'w') as f:
    json.dump(posts_list, f, indent = 4, cls=NumpyEncoder)

def create_env_fb(Saving_path, id = 686): # Scale Free 20
  if not os.path.exists(Saving_path):
    os.makedirs(Saving_path, exist_ok=True)
  else:
    shutil.rmtree(Saving_path)
    os.makedirs(Saving_path, exist_ok=True)

  # Define each agent and his/her/their relations
  env_create_agent_test_facebook(Saving_path, id = id)

  # Create list of rumors and posts
  rumor_list = create_rumors_test()
  with open(Saving_path+f'/rumor_list.json', 'w') as f:
    json.dump(rumor_list, f, indent = 4, cls=NumpyEncoder)

  posts_list = create_random_posts_test(50)
  with open(Saving_path+f'/posts_list.json', 'w') as f:
    json.dump(posts_list, f, indent = 4, cls=NumpyEncoder)
  

def main():
  Code_dir_path = 'path_to_multi-agent-framework/multi-agent-framework/'  # Put the current code directory path here
  Saving_path = Code_dir_path + 'Env_Rumor_Test'
  
  # The first time to create the environment, after that you can comment it
  create_env2(Saving_path)
  #G = read_facebook_network(686)
  #create_env_fb(Saving_path, 686)

if __name__ == "__main__":
  main()