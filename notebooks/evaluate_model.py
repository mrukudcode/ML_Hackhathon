# evaluate_model.py

import os
import torch
import joblib
import numpy as np

# IMPORTANT: We must import the classes and constants from the training script
# so the evaluation knows the network size and structure.
from Q_learning_agent import (
    DQNAgent, 
    evaluate, 
    load_hmm_model,
    INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, 
    LR, 
    MODEL_PATH,
    ALPHABET,
    MAX_WRONG,
    GAMMA
)

# NOTE: The evaluate function in Q_learning_agent.py needs the global 'hmm' object 
# to run 'vectorize_state'. We must ensure it's loaded here.
try:
    global hmm
    hmm = load_hmm_model()
    # The vectorize_state function in Q_learning_agent.py will now use this global hmm instance.
except Exception as e:
    print(f"❌ Critical Error: Could not load HMM model (hmm_model.joblib). Have you run python train_hmm.py?")
    exit()

def run_evaluation():
    """Initializes the agent, loads the saved weights, and runs evaluation."""
    
    # 1. Check for Saved Model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model weights not found at '{MODEL_PATH}'.")
        print("Please ensure 'dqn_model.pth' exists (requires at least partial training).")
        return
    
    # 2. Initialize Agent Structure
    # Use dummy optimizer params, as we only need the network structure for loading
    optimizer_params = {'lr': LR} 
    # Must pass the correct network dimensions
    agent = DQNAgent(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, optimizer_params, gamma=GAMMA)
    
    # 3. Load Trained Weights
    print(f"Loading trained DQN weights from: {MODEL_PATH}")
    agent.q_network.load_state_dict(torch.load(MODEL_PATH))
    agent.q_network.eval() # Set network to evaluation mode (no dropout, etc.)

    # 4. Run Evaluation
    print("Starting evaluation on test set (greedy mode)...")
    # 'greedy=True' ensures the agent only exploits its knowledge (epsilon=0.0)
    evaluate(agent, test_path="test.txt", max_wrong=MAX_WRONG, greedy=True)

if __name__ == "__main__":
    run_evaluation()