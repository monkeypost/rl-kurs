import gymnasium as gym
import numpy as np

np.set_printoptions(linewidth=np.inf)
env = gym.make('CliffWalking-v0')

env.reset()

states = []
policy = []
for i in range(48):
    states.append(0)
    policy.append(0)

gamma = 0.8

def policy_evaluation():
    #Policy Evalution
    steps = 0
    while True:
        threshold = 0.00001 #Taken from 100 iterations
        delta = 0 #With help from ChatGpt 
        for s in range(48):       
            expected_value = 0
            current_value = states[s]
            for a in (range(4)):
                s_prime = env.unwrapped.P[s][a][0][1]
                r = env.unwrapped.P[s][a][0][2]
                terminal = env.unwrapped.P[s][a][0][3]
                
                if (r == -100): #Without this the agent gets trapped in left corner
                    r = -1 

                if (terminal):
                    r = 10
                expected_value += 0.25 * (r + gamma * states[s_prime])        
            delta = max(delta, abs(current_value - expected_value)) #With help from ChatGpt  
            states[s] = round(expected_value, 5)
        steps += 1
        print(f"Iteration: {steps}")        
        print(np.reshape(states, (4,12)))
        if (delta < threshold): #With help from ChatGpt 
            break

#Policy improvement
while True:
    policy_evaluation()
    policy_stable = True

    for s in range(len(policy)):
        old_action = policy[s]
        
        for a in (range(4)):
            min_value = states[s]
            if states[env.unwrapped.P[s][a][0][1]] > min_value:
                policy[s] = a
        if old_action != policy[s]:
            policy_stable = False
    if policy_stable:
        break

print(np.reshape(policy, (4,12)))

#Run policy
done = False

env = gym.make('CliffWalking-v0', render_mode="human")

observation = env.reset()
observed_state = observation[0]

while done != True:
    env.render()
    
    action = policy[observed_state]
    observation, reward, done, truncated, info = env.step(action)
    observed_state = observation

env.close()