# SARSA Learning Algorithm


## AIM
Write the experiment AIM.

## PROBLEM STATEMENT
Explain the problem statement.

## SARSA LEARNING ALGORITHM
Include the steps involved in the SARSA Learning algorithm

## SARSA LEARNING FUNCTION
### Name: D Vergin Jenifer
### Register Number: 212223240174

### SARSA Learning function:

```
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)
    for e in tqdm(range(n_episodes), leave=False):
      state, done = env.reset(), False
      action = select_action(state, Q, epsilons[e])
      while not done:
        next_state, reward, done, _ = env.step(action)
        next_action = select_action(next_state, Q, epsilons[e])
        td_target=reward + gamma * Q[next_state][next_action] * (not done)
        td_error = td_target - Q[state][action]
        Q[state][action] = Q[state][action] + alphas[e] * td_error
        state, action = next_state, next_action
      Q_track[e] = Q
      pi_track.append(np.argmax(Q,axis=1))
    V=np.max(Q,axis=1)
    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:
<img width="692" height="336" alt="image" src="https://github.com/user-attachments/assets/3033c76e-c2f8-4e51-bd69-1951b6a30e13" />
<img width="450" height="112" alt="image" src="https://github.com/user-attachments/assets/2ab282e7-2d97-40ba-a206-18c0c6857a41" />
<img width="985" height="712" alt="image" src="https://github.com/user-attachments/assets/af1954a0-7597-4f2f-912f-cab79afac505" />
<img width="683" height="163" alt="image" src="https://github.com/user-attachments/assets/8d3352f3-18a7-4e38-83ea-d8ef7111eb6e" />

Include plot comparing the state value functions of Monte Carlo method and SARSA learning.

## RESULT:

Write your result here
