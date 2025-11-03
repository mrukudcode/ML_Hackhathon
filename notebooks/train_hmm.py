# train_hmm.py

import os
# Import the corrected HMM class from your simple_hmm.py file
from simple_hmm import SimpleHMM 

def run_hmm_training(corpus_path="corpus.txt", model_path="hmm_model.joblib"):
    """
    Initializes, trains, and saves the HMM model.
    Assumes 'corpus.txt' is in the same directory.
    """
    if not os.path.exists(corpus_path):
        # Fallback for common project structures
        corpus_path = os.path.join("Data", "corpus.txt")
        if not os.path.exists(corpus_path):
            print(f"❌ Error: Corpus file not found at 'corpus.txt' or 'Data/corpus.txt'.")
            return

    print("Starting HMM training...")
    hmm = SimpleHMM()
    hmm.train(corpus_path)
    hmm.save(model_path)
    print("HMM training complete. Ready to run DQN agent.")

if __name__ == "__main__":
    run_hmm_training()