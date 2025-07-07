import numpy as np
import matplotlib.pyplot as plt
import random

# 보수 행렬, C: cooperate, D: Decieve
payoffs = {
    ('C', 'C'): (3, 3),
    ('C', 'D'): (0, 5),
    ('D', 'C'): (5, 0),
    ('D', 'D'): (1, 1)
}

ACTIONS = ['C', 'D']
STATE_SPACE = [(a1, a2) for a1 in ACTIONS for a2 in ACTIONS] + [('None', 'None')]

# Q-learning 에이전트
class QLearningAgent:
    def __init__(self, name, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.name = name
        self.q_table = {s: {a: 0.0 for a in ACTIONS} for s in STATE_SPACE}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.reset()
    
    def reset(self):
        self.last_state = ('None', 'None')

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        return max(self.q_table[state], key=self.q_table[state].get)

    def update(self, state, action, reward, next_state):
        best_next = max(self.q_table[next_state], key=self.q_table[next_state].get)
        td_target = reward + self.gamma * self.q_table[next_state][best_next]
        self.q_table[state][action] += self.alpha * (td_target - self.q_table[state][action])

# 고정 전략 에이전트
class FixedStrategyAgent:
    def __init__(self, name, strategy_fn):
        self.name = name
        self.strategy_fn = strategy_fn
        self.reset()

    def reset(self):
        self.history = []
        self.opponent_history = []

    def select_action(self, _):
        return self.strategy_fn(self.history, self.opponent_history)

    def update(self, my_action, opponent_action):
        self.history.append(my_action)
        self.opponent_history.append(opponent_action)

# 고정 전략 함수들
def always_cooperate(history, opponent_history):
    return 'C'

def always_defect(history, opponent_history):
    return 'D'

def random_strategy(history, opponent_history):
    return random.choice(['C', 'D'])

def grim_trigger(history, opponent_history):
    if 'D' in opponent_history:
        return 'D'
    return 'C'

# 의사 비전단 확률
def apply_noise(move, error_rate):
    return 'D' if move == 'C' and random.random() < error_rate else \
           'C' if move == 'D' and random.random() < error_rate else move

# 한 게임 수행
def play_episode(agent1, agent2, rounds=100, error_rate=0.1, training=True):
    state1 = state2 = ('None', 'None')
    total_r1 = total_r2 = 0
    scores = []

    agent1.reset()
    agent2.reset()

    for _ in range(rounds):
        a1 = agent1.select_action(state1)
        a2 = agent2.select_action(state2)

        a1_seen = apply_noise(a1, error_rate)
        a2_seen = apply_noise(a2, error_rate)

        r1, r2 = payoffs[(a1, a2)]
        total_r1 += r1
        total_r2 += r2
        scores.append((r1 + r2) / 2)

        next_state1 = (a1_seen, a2_seen)
        next_state2 = (a2_seen, a1_seen)

        # Qlearning Studying
        if training and isinstance(agent1, QLearningAgent):
            agent1.update(state1, a1, r1, next_state1)
        if training and isinstance(agent2, QLearningAgent):
            agent2.update(state2, a2, r2, next_state2)

        # Fixed --> Upadate 
        if isinstance(agent1, FixedStrategyAgent):
            agent1.update(a1, a2)
        if isinstance(agent2, FixedStrategyAgent):
            agent2.update(a2, a1)

        state1 = next_state1
        state2 = next_state2

    return total_r1, total_r2, scores

# Creating agent:TODO
def create_agents():
    agents = {
        'TFT_Agent': QLearningAgent('TFT_Agent'),
        'QLearner_1': QLearningAgent('QLearner_1'),
        'Grim_Agent': FixedStrategyAgent('Grim_Agent', grim_trigger),
        'AlwaysCoop': FixedStrategyAgent('AlwaysCoop', always_cooperate),
        'AlwaysDefect': FixedStrategyAgent('AlwaysDefect', always_defect),
        'RandomAgent': FixedStrategyAgent('RandomAgent', random_strategy)
    }
    return agents

# Studying
def train_all(agents, epochs=400, rounds=100, error_rate=0.1):
    names = list(agents.keys())
    score_history = []

    for epoch in range(epochs):
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a1 = agents[names[i]]
                a2 = agents[names[j]]
                _, _, scores = play_episode(a1, a2, rounds, error_rate, training=True)
                score_history.append(np.mean(scores))
    return score_history

# Evoaluation
def evaluate_all(agents, rounds=400, error_rate=0.1):
    names = list(agents.keys())
    n = len(names)
    score_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            a1 = agents[names[i]]
            a2 = agents[names[j]]
            r1, r2, _ = play_episode(a1, a2, rounds, error_rate, training=False)
            score_matrix[i, j] = r1 / rounds
    return names, score_matrix

# Playing
agents = create_agents()
match_score_history = train_all(agents, epochs=400, rounds=100, error_rate=0.1)
agent_names, score_matrix = evaluate_all(agents, rounds=400, error_rate=0.1)

error_rates = np.linspace(0.1, 0.9, 9)
heatmaps = []

for error in error_rates:
    agents = create_agents()
    train_all(agents, epochs=400, rounds=100, error_rate=error)
    names, score_matrix = evaluate_all(agents, rounds=400, error_rate=error)
    heatmaps.append(score_matrix)

# Plot heatmaps for each error rate
fig, axes = plt.subplots(3, 3, figsize=(16, 14))
axes = axes.flatten()

for idx, error in enumerate(error_rates):
    ax = axes[idx]
    im = ax.imshow(heatmaps[idx], cmap="coolwarm", vmin=0, vmax=5)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{heatmaps[idx][i, j]:.1f}", ha="center", va="center", color="black")
    ax.set_title(f"Error Rate: {int(error * 100)}%")
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)

fig.suptitle("Average Score Between Agents (After Training) at Different Error Rates", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()