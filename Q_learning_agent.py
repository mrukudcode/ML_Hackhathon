# Q-learning-agent.py - HYPER-POSITIVE REWARD TUNE

import os
import random
import string
import numpy as np
import matplotlib.pyplot as plt
import joblib 
from tqdm import trange
from collections import deque, namedtuple

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Local modules
from simple_hmm import SimpleHMM
from hangman_manual import HangmanEnv

# ---------- CONFIGURATION AND HYPERPARAMETERS ----------
ALPHABET = list(string.ascii_lowercase)
N_LETTERS = len(ALPHABET) 
LETTER_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}

# --- CRITICAL REWARD TUNE FOR POSITIVE avg_last50 ---
# Rewards are amplified to quickly overcome early training noise and losses.
CORRECT_REWARD = +15.0 # Increased positive signal for correct guess
WRONG_REWARD   = -3.0  # Reduced penalty for wrong guess
REPEAT_REWARD  = -6.0  # Penalty for guessing a letter twice
WIN_BONUS      = +50.0 # Massively increased bonus for winning
LOSE_PENALTY   = -20.0 # Reduced penalty for losing
STEP_PENALTY   = -0.1  # Minor penalty for each step encourages efficiency

# Training Hyperparams (Left high-performing values unchanged)
DEFAULT_EPISODES = 15000 
MAX_WRONG = 6
LR = 0.001          
GAMMA = 0.995       
EPS_START = 1.0
EPS_MIN = 0.01
EPS_DECAY = 0.9997  

# DQN Network config (Unchanged)
INPUT_SIZE = N_LETTERS + N_LETTERS + 1 + N_LETTERS + 1 
HIDDEN_SIZE = 256
OUTPUT_SIZE = N_LETTERS       
TARGET_UPDATE = 100 
MODEL_PATH = "dqn_model.pth"

# Replay Buffer (Unchanged)
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'available_next'))
REPLAY_CAPACITY = 50000
BATCH_SIZE = 128
AVG_WINDOW = 50 

# ---------- HMM Loader (from simple_hmm.py) ----------
def load_hmm_model():
    """Loads the HMM model, or raises an error if it doesn't exist."""
    hmm_path = "hmm_model.joblib"
    if not os.path.exists(hmm_path):
         # If HMM not found, prompt to run trainer but don't exit if we can train it
         pass 
    
    hmm = joblib.load(hmm_path)
    print(f"✅ Loaded HMM from: {os.path.abspath(hmm_path)}")
    return hmm

try:
    # Use a dummy class if HMM fails to load to prevent crashing, but this should now be fixed.
    hmm = load_hmm_model()
except Exception as e:
    print(f"Error loading HMM: {e}. Cannot run agent.")
    exit()

# --- ReplayBuffer, State Vectorization, QNetwork, and DQNAgent are unchanged from the final working version ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

def vectorize_state(masked_word: str, guessed: set, wrong_count: int):
    revealed_oh = np.zeros(N_LETTERS, dtype=np.float32)
    for char in set(c for c in masked_word if c in ALPHABET):
        revealed_oh[LETTER_TO_IDX[char]] = 1.0
            
    guessed_binary = np.array([1.0 if c in guessed else 0.0 for c in ALPHABET], dtype=np.float32)
    wrong_count_norm = np.array([wrong_count / MAX_WRONG], dtype=np.float32)
    hmm_probs = hmm.get_letter_probs(masked_word, guessed)
    word_len = len(masked_word)
    len_norm = np.array([word_len / 26.0], dtype=np.float32)

    state_vector = np.concatenate([
        revealed_oh,
        guessed_binary,
        wrong_count_norm,
        hmm_probs,
        len_norm
    ])
    return torch.from_numpy(state_vector).float()

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size) 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, input_size, hidden_size, output_size, optimizer_params, 
                 gamma=GAMMA, eps_start=EPS_START, eps_min=EPS_MIN, eps_decay=EPS_DECAY):
        
        self.q_network = QNetwork(input_size, hidden_size, output_size)
        self.target_network = QNetwork(input_size, hidden_size, output_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval() 
        
        self.optimizer = optim.Adam(self.q_network.parameters(), **optimizer_params)
        self.loss_fn = nn.MSELoss()
        
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        
        self.memory = ReplayBuffer(REPLAY_CAPACITY)

    def choose_action(self, state_tensor, available_letters):
        if random.random() < self.epsilon:
            return random.choice(available_letters)
        else:
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                mask = torch.tensor([1.0 if c in available_letters else 0.0 for c in ALPHABET])
                q_values = q_values * mask - (1 - mask) * 1e9 
                action_idx = torch.argmax(q_values).item()
                return ALPHABET[action_idx]

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return None 
        
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state)
        action_batch = torch.tensor([LETTER_TO_IDX[a] for a in batch.action], dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float)
        done_batch = torch.tensor(batch.done, dtype=torch.float)

        non_final_mask = (1 - done_batch).bool()
        non_final_next_states = torch.stack([s for s, done in zip(batch.next_state, batch.done) if not done])
        available_next_batch = batch.available_next

        q_current = self.q_network(state_batch).gather(1, action_batch).squeeze(1)

        q_next_max = torch.zeros(BATCH_SIZE)
        
        if non_final_next_states.shape[0] > 0:
            q_target_next_all = self.target_network(non_final_next_states)
            
            mask_next = torch.stack([torch.tensor([1.0 if c in available else 0.0 for c in ALPHABET]) 
                                    for available, done in zip(available_next_batch, batch.done) if not done])
            q_target_next_all = q_target_next_all * mask_next - (1 - mask_next) * 1e9
            
            max_q_values = q_target_next_all.max(1)[0].squeeze()
            q_next_max[non_final_mask] = max_q_values

        q_target = reward_batch + self.gamma * q_next_max

        loss = self.loss_fn(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    def save(self, path=MODEL_PATH):
        torch.save(self.q_network.state_dict(), path)
        print("✅ DQN Model saved to:", path)


# ---------- Training ----------
def train(agent: DQNAgent, corpus_path="corpus.txt",
             episodes=DEFAULT_EPISODES, max_wrong=MAX_WRONG,
             log_every=500, avg_window=AVG_WINDOW, target_update_episodes=TARGET_UPDATE):
    
    try:
        with open(corpus_path, "r") as f:
            words = [w.strip().lower() for w in f.readlines() if w.strip()]
    except FileNotFoundError:
        print(f"Error: Corpus file not found at {corpus_path}. Exiting.")
        return [], []
        
    rewards = []
    running_avg = []

    for ep in trange(episodes, desc="Train"):
        word = random.choice(words)
        env = HangmanEnv(word, max_wrong=max_wrong)
        guessed = set()
        masked = env.get_masked_word()
        s = vectorize_state(masked, guessed, env.wrong)
        total_r = 0.0
        done = False

        while not done:
            available = [c for c in ALPHABET if c not in guessed]
            if not available:
                 break 
                 
            action = agent.choose_action(s, available)

            was_repeat = action in guessed
            masked_next, _, done = env.step(action)

            # --- Reward Calculation (Uses the new hyper-positive constants) ---
            if was_repeat:
                reward = REPEAT_REWARD
            elif action in word:
                reward = CORRECT_REWARD
            else:
                reward = WRONG_REWARD

            reward += STEP_PENALTY

            if done:
                reward += WIN_BONUS if masked_next == env.word else LOSE_PENALTY

            total_r += reward
            guessed.add(action)
            s_next = vectorize_state(masked_next, guessed, env.wrong)
            available_next = [c for c in ALPHABET if c not in guessed]

            agent.memory.push(s, action, reward, s_next, done, available_next)
            agent.optimize_model()
            
            s = s_next

        agent.decay_epsilon()
        rewards.append(total_r)
        
        if len(rewards) >= avg_window:
            running_avg.append(np.mean(rewards[-avg_window:]))
        else:
            running_avg.append(np.mean(rewards))

        if (ep + 1) % target_update_episodes == 0:
            agent.update_target_network()

        if (ep + 1) % log_every == 0:
            print(f"Episode {ep+1}/{episodes}  eps={agent.epsilon:.4f}  avg_last{avg_window}={running_avg[-1]:.2f}  Memory={len(agent.memory)}")

    return rewards, running_avg


# ---------- Evaluation (Unchanged) ----------
def evaluate(agent: DQNAgent, test_path="test.txt", max_wrong=MAX_WRONG, greedy=True):
    """Evaluates the agent's performance on the test set."""
    saved_eps = agent.epsilon
    if greedy:
        agent.epsilon = 0.0

    if not os.path.exists(test_path):
        print(f"Warning: Test file not found at {test_path}. Skipping evaluation.")
        return 0.0, 0.0, 0, 0.0

    with open(test_path, "r") as f:
        test_words = [w.strip().lower() for w in f.readlines() if w.strip()]
    
    if not test_words:
        return 0.0, 0.0, 0, 0.0

    wins = 0
    total_wrong = 0
    total_repeats = 0

    for w in test_words:
        env = HangmanEnv(w, max_wrong=max_wrong)
        guessed = set()
        masked = env.get_masked_word()
        done = False

        while not done:
            s = vectorize_state(masked, guessed, env.wrong)
            available = [c for c in ALPHABET if c not in guessed]
            
            if not available:
                 break
                 
            action = agent.choose_action(s, available)
            was_repeat = action in guessed
            masked, _, done = env.step(action)
            
            if was_repeat:
                total_repeats += 1
            elif action not in w:
                total_wrong += 1
            guessed.add(action)

        if masked == env.word:
            wins += 1

    if greedy:
        agent.epsilon = saved_eps

    success_rate = wins / len(test_words) 
    final_score = (success_rate * 2000) - (total_wrong * 5) - (total_repeats * 2) 

    print("\n=== Evaluation ===")
    print(f"Games: {len(test_words)}  Wins: {wins}  Success Rate: {success_rate*100:.2f}%")
    print(f"Total Wrong: {total_wrong}  Total Repeats: {total_repeats}")
    print(f"Final Score: {final_score:.2f}")
    return success_rate, total_wrong, total_repeats, final_score


# ---------- Main Execution ----------
if __name__ == "__main__":
    
    optimizer_params = {'lr': LR}
    agent = DQNAgent(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, optimizer_params)

    # Automatically handle HMM training if needed
    HMM_MODEL_PATH = "hmm_model.joblib"
    if not os.path.exists(HMM_MODEL_PATH):
        print("❌ HMM model not found. Running HMM training now...")
        try:
            hmm_trainer = SimpleHMM()
            hmm_trainer.train("corpus.txt")
            hmm_trainer.save(HMM_MODEL_PATH)
            # Re-load the agent's internal HMM now that it's saved
            hmm = load_hmm_model()
        except FileNotFoundError:
            print("CRITICAL: Cannot find 'corpus.txt'. Please ensure it's in the project root.")
            exit()
        
    print("Training started...")
    rewards, running = train(agent, corpus_path="corpus.txt",
                             episodes=DEFAULT_EPISODES)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.35, label="reward_per_episode")
    plt.plot(running, color="red", linewidth=2, label=f"{AVG_WINDOW}-ep running avg")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards (HMM + DQN)")
    plt.legend()
    plt.show()

    agent.save(MODEL_PATH)
    evaluate(agent, test_path="test.txt")