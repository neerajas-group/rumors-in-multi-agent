# Simulating Rumor Spreading in Social Networks using LLM agents
### Rumors-in-Multi-Agent-Simulation
This repository contains the code for *WMAC 2025: AAAI 2025 Workshop on Advancing LLM-Based Multi-Agent Collaboration* Paper: **Simulating Rumor Spreading in Social Networks using LLM agents**. The code is inspired by the paper: **MIT-REALM-Multi-Robot** (https://yongchao98.github.io/MIT-REALM-Multi-Robot/).

## Requirements
Install the required Python packages using the following command:
pip install numpy openai re random time copy tiktoken networkx

Alternatively, you can install them using requirements.txt.

Additionally, obtain your OpenAI API key from https://beta.openai.com/. Add the key (starting with 'sk-') to LLM.py on line 8.

The Facebook Social Network dataset can be downloaded from https://snap.stanford.edu/data/ego-Facebook.html.

## Create Testing Environments
Generate a network and assign agents to each node as the first step.

Run rumor_test_env.py to create the environments. The script supports three types of networks and the Facebook dataset. The following functions are used to generate specific network types:
- create_env1: Random network
- create_env2: Scale-free network
- create_env3: Small-world network
- create_env_fb: Facebook dataset-based network

For the first three network types, the number of nodes and agent configurations can be modified in their respective generation functions. Use the add_fact_check parameter to enable an agent-based fact checker that connects to all nodes.

To use a specific create_env* function, add it to the main function in the script. Note: Creating a new environment will overwrite any existing one.

Run the following command to generate the environment:
python rumor_test_env.py


## Usage
Run rumor_test_run.py to simulate rumor spreading in social networks. Modify the models (e.g., GPT-4o, GPT-4o-mini) around line 265.

Adjustable parameters:
- query_time_limit: Number of iterations.
- agent_count: Number of agents (must match the value used during environment creation).
- num_of_initial_posts: Number of random posts initially assigned to each agent.
- selection_policy: Agent selection strategy ('random' or 'mff', where 'mff' gives preference to agents with more friends).
- patient_zero_policy: Rumor initialization strategy ('random' or 'mff', where 'mff' starts with the agent with the most friends).
- fact_checker: Whether to include a fact checker (options: None, 'agent-based', 'special').
- fact_checker_freq: Frequency of fact-checker activation.
- filter_friends: Percentage of friends who will not receive a post.
- filter_post: Probability of a post being deleted.

Run the following command to start the simulation:
python rumor_test_run.py


The experimental results and rumor matrix will be saved in the specified path (path_to_multi-agent-framework).

## Additional Files
- agents_XXX.json: Defines agent personas.
- LLM.py: Provides ChatGPT API access.
- prompt_rumor_test.py: Contains prompts and supporting functions.
- plotting_scripts/: Includes scripts for generating visualizations for M1/M2/M3.
