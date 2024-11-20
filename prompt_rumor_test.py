from LLM import *
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
enc = tiktoken.encoding_for_model("gpt-4")
input_prompt_token_limit = 3000

likely_to_accept_rumors = {
    '1': "won't easily accept any rumors or new information unless they are confirmed or well-examined",
    '2': "may suspect rumors but will accept them once they appear frequently in posts or generally make sense",
    '3': "will accept any new information unless there is significant controversy or criticism",
    '4': "will easily accept any rumors, even if there are doubts or criticisms",
}

likely_to_forward_rumors = {
    '1': "prefer not to spread much of the new information seen in others' posts",
    '2': "may forward posts seen with comments and feelings, or may just share personal experiences",
    '3': "are willing to share and comment on rumors, posts, and new things seen in posts",
}

def LLM_summarize_func(state_action_prompt_next_initial, model_name):
  prompt1 = f"Please summarize the following content as concise as possible: \n{state_action_prompt_next_initial}"
  messages = [{"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": prompt1}]
  response = GPT_response(messages, model_name)
  return response

'''
Inputs:
  Agent Info: agent_name, agent_age, agent_job, agent_traits
  Acceptance to and willingness to spread information:
      agent_rumors_acc : 1 - 4
      agent_rumors_spread : 1 - 3

  friend_list: list of friends (nodes)
  post_history: complete history of posts visible to the agent.
  rumor_list: list of all rumors
  
  dialogue_history_method: way to record history
'''
def input_prompt_local_agent_DMAS_dialogue_func(agent_name, agent_age, agent_job, agent_traits,
                                                friend_list,
                                                agent_rumors_acc, agent_rumors_spread,
                                                post_history, rumor_list, rumor_believe_list,
                                                dialogue_history_method):
#  It can be something you've read in other posts,
#  something about your job or day, or a simple greeting.
  user_prompt_1 = f'''
  Hi, {agent_name}, you are a {agent_age}-year-old {agent_job} known for being {agent_traits}. Please follow the instructions below.
  You are active on a social network, receiving and sending posts. 
  You {likely_to_accept_rumors[agent_rumors_acc]}, and you {likely_to_forward_rumors[agent_rumors_spread]}.
  Read through the post history, especially the new posts. It can be something you've read in other posts but you need to rephase it your personality.
  You can criticize the posts if you don't agree with them, you can also repeat them or express in your own way.
  Your posts can be seen by all your friends. Here are your friends: {{{friend_list}}}
  You are about to send a new post [POST] based on your personal preferences. 


  After posting, you will review a list of rumors and decide [CHECK] whether to believe or reject each one. Be honest: if your post mentions a rumor,
  your response must be consistent with what you posted.

  [Action Output Instruction]
  Start with 'POST', then on a new line, specify the content of your new post.
  Then, on a new line, output 'CHECK', followed by True or False for each rumor.
  Example#1: 
  POST
  I just read that Donald Trump will be president of Greece! OMG! That's interesting.
  CHECK
  False COVID-19 now named as COVID-114514.
  True Donald Trump will be president of Greece.

  Example#2: 
  POST
  What a nice day! I enjoy my job as a teacher.
  CHECK
  False COVID-19 now named as COVID-114514.
  False Donald Trump will be president of Greece.
  
  '''

  if sum(rumor_believe_list) == 0:
    pass
  else:
    user_prompt_1 += 'Before you reviewing the posts, you used to believe:\n'
    for i in range(len(rumor_believe_list)):
      if rumor_believe_list[i]:
        user_prompt_1 += f'You used to believe {rumor_list[str(i)]} is True\n'

  user_prompt_1 += f'''
  The previous post history is: {{{post_history}}}
  Think step-by-step about the task. Be careful not to let the rumor list affect your judgment on post history.
  You CANNOT post the information from the rumor list but NOT in your post history.
  The rumor list is: {{{rumor_list}}} Check whether you believe them based on what you read and send.
  Try not to exactly repeat what others have said.
  Propose exactly one action (POST and CHECK) for yourself in the current round.

  Your response:
  '''
  token_num_count = len(enc.encode(user_prompt_1))
  print(f'Number of Token in the request: {token_num_count}')

  if dialogue_history_method == 'all_history':
    pass

  return user_prompt_1

def message_construct_func(user_prompt_list, response_total_list, dialogue_history_method):
  if f'{dialogue_history_method}' == 'TODO_2_history':
    messages=[{"role": "system", "content": "You are a helpful assistant."}]
    #print('length of user_prompt_list', len(user_prompt_list))
    for i in range(len(user_prompt_list)):
      messages.append({"role": "user", "content": user_prompt_list[i]})
      if i < len(user_prompt_list)-1:
        messages.append({"role": "assistant", "content": response_total_list[i]})
    #print('Length of messages', len(messages))
  elif f'{dialogue_history_method}' in ('TODO_history', 'all_history'):
    messages=[{"role": "system", "content": "You are a helpful assistant."}]
    messages.append({"role": "user", "content": user_prompt_list[-1]})
    #print('Length of messages', len(messages))
  return messages