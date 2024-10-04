from LLM import *
from rumor_test_env import *
from prompt_rumor_test import *
from sre_constants import error
import random
import os
import json
import re
import copy
import numpy as np
import shutil
import time

def analyze_input(text):
    # Using \s+ to handle multiple whitespaces and \s* to allow optional whitespace
    post_pattern = re.compile(r'POST\s+(.*?)(?=\s+CHECK)', re.DOTALL)
    check_pattern = re.compile(r'CHECK\s+(.*)', re.DOTALL)

    # Find the POST section
    post_match = post_pattern.search(text)
    if not post_match:
        raise ValueError("POST section not found or incorrectly formatted.")
    post = post_match.group(1).strip()
    if not post:
        raise ValueError("POST content is empty.")

    # Find the CHECK section
    check_match = check_pattern.search(text)
    if not check_match:
        raise ValueError("CHECK section not found or incorrectly formatted.")

    # Split CHECK section into lines and parse them
    check_lines = check_match.group(1).strip().split('\n')
    check_list = []
    for line in check_lines:
        if not re.match(r'^(True|False)', line):
            raise ValueError("Lines after CHECK must start with True or False.")
        # Parse boolean value
        check_list.append(line.startswith('True'))

    return post, check_list

def run_exp(Saving_path, iteration_num, query_time_limit, agent_count, num_of_initial_posts, dialogue_history_method='_w_only_state_action_history', cen_decen_framework='DMAS', model_name = 'gpt-4o'):
    
    agent_dir = Saving_path
    agent_list = []

    # Read rumor list
    with open(Saving_path+f'/rumor_list.json', 'r') as f:
        rumor_list = json.load(f)
    # Read other posts list
    with open(Saving_path+f'/posts_list.json', 'r') as f:
        posts_list = json.load(f)

    for i in range(agent_count):
        with open(Saving_path+f'/agent_{i}/agent_{i}.json', 'r') as f:
            agent_list.append(json.load(f))

    # Initialization of rumor list
    rumor_matrix = [[False for _ in range(len(rumor_list))] for __ in range(agent_count)]
    print(rumor_matrix)
    # All posts list
    #beginning_list = {k: (rumor_list.get(k, '') + posts_list.get(k, '')).strip() for k in sorted(set(rumor_list) | set(posts_list))}    
    post_history = ['' for _ in range(agent_count)]
    post_count = [0 for _ in range(agent_count)]

    # Assign rumors to random agent's history
    rumor_list_copy = rumor_list.copy()
    while rumor_list_copy:
        random_agent = random.randint(0, agent_count-1)
        random_key = random.choice(list(rumor_list_copy.keys()))
        random_rumor = rumor_list_copy.pop(random_key)
        post_history[random_agent] += f'Random post: {random_rumor}\n'
        post_count[random_agent] += 1

        print(f'Rumor {random_key}: {random_rumor} is assigned to Agent {random_agent}')

    # Assign random contents to agents
    for i in range(agent_count):
        while post_count[i] < num_of_initial_posts:
            random_key = random.choice(list(posts_list.keys()))
            post_history[i] += f'Random post: {posts_list[random_key]}\n'
            post_count[i] += 1

        print(f'Initialization for agent {i}, info: {agent_list[i]}\n Initial {post_count[i]} posts: {post_history[i]}')
        
    print('Initialization done!')

    # for query_time_limit of timestamps
    for ts in range(query_time_limit):
        print(f'=================================分割线==============================')
        print(f'Timestamp {ts}')
        # Picks an agent at random
        i = random.randint(0, agent_count-1)
        print(f'Pick agent {i} to act')
        ag = agent_list[i]
        # Step 1: Generate response using ChatGPT api
        prompt = input_prompt_local_agent_DMAS_dialogue_func(ag['agent_name'], ag['agent_age'], ag['agent_job'], 
                                                                ag['agent_traits'], ag['friends'],
                                                                ag['agent_rumors_acc'], ag['agent_rumors_spread'],
                                                                post_history[i], rumor_list,
                                                                dialogue_history_method)
        print(f'Feeding prompt to ChatGPT: \n{prompt}')

        # Process message
        messages=[{"role": "system", "content": "You are a helpful assistant."}]
        messages.append({"role": "user", "content": prompt})

        initial_response, token_num_count = GPT_response(messages,model_name)
        print(f'Getting response from ChatGPT: \n{initial_response}')
        # Step 2: Post-process the requests
        post, check_list = analyze_input(initial_response)
        # Step 2a: Append History
        print(f'Appending post: {post} to history')
        post_history[i] += f"{ag['agent_name']}: {post}\n"
        for friend in ag['friends']:
            print(f'Update to friend {friend}')
            post_history[friend] += f"{ag['agent_name']}: {post}\n"

        # Step 2b: Fact Check
        for ru in range(len(rumor_list)):
            print(f"Agent {i} {ag['agent_name']} believes rumor {ru} {rumor_list[str(ru)]} is {check_list[ru]}")
        rumor_matrix[i] = check_list

    return rumor_matrix

random.seed(42) #萬世一系ノ宇宙ノ真理ノ種

Code_dir_path = 'path_to_multi-agent-framework/multi-agent-framework/' # Put the current code directory path here
Saving_path = Code_dir_path + 'Env_Rumor_Test'
model_name = 'gpt-4o'  #'gpt-4-0613', 'gpt-3.5-turbo-16k-0613' # 4o should be fine
print(f'-------------------Model name: {model_name}-------------------')

query_time_limit = 10
iterations = 1
agent_count = 3
num_of_initial_posts = 2

for iteration_num in range(iterations):
    print('-------###-------###-------###-------')
    print(f'Iteration num is: {iteration_num}\n\n')
    #user_prompt_list, response_total_list, pg_state_list, success_failure, index_query_times, token_num_count_list, Saving_path_result 
    rumor_matrix= run_exp(Saving_path, iteration_num, query_time_limit, agent_count, num_of_initial_posts, dialogue_history_method='_w_only_state_action_history',
            #cen_decen_framework='HMAS-2', model_name = model_name)
            cen_decen_framework='DMAS', model_name = model_name)

    print(f'Done')
    '''
    with open(Saving_path_result + '/token_num_count.txt', 'w') as f:
        for token_num_num_count in token_num_count_list:
        f.write(str(token_num_num_count) + '\n')

    with open(Saving_path_result + '/success_failure.txt', 'w') as f:
        f.write(success_failure)

    with open(Saving_path_result + '/env_action_times.txt', 'w') as f:
        f.write(f'{index_query_times+1}')
    print(success_failure)
    print(f'Iteration number: {index_query_times+1}')
    '''

