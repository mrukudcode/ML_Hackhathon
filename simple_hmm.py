# simple_hmm.py

import numpy as np
import string
from collections import Counter, defaultdict
import joblib

class SimpleHMM:
    """
    A Bigram Hidden Markov Model for calculating letter probabilities 
    based on the surrounding characters (left and right context).
    """
    def __init__(self):
        # Stores counts of bigrams (e.g., 'a' -> 't' count)
        self.bigram_counts = defaultdict(Counter)
        # Stores counts of unigrams (denominator for transition probability)
        self.unigram_counts = Counter()
        self.vocab = list(string.ascii_lowercase)

    def train(self, corpus_path):
        """Trains the HMM on the word corpus."""
        # Note: corpus_path should be 'Data/corpus.txt' if using your structure
        with open(corpus_path, 'r') as f:
            words = [w.strip().lower() for w in f.readlines() if w.strip()]
        
        for word in words:
            # '<s>' represents the start-of-word token
            prev = "<s>"
            for ch in word:
                if ch in self.vocab:
                    self.bigram_counts[prev][ch] += 1
                    self.unigram_counts[prev] += 1
                    prev = ch
            # '</s>' represents the end-of-word token
            self.bigram_counts[prev]["</s>"] += 1
            self.unigram_counts[prev] += 1 # Count the transition from the last letter

    def get_letter_probs(self, masked_word, guessed):
        """
        Calculates the probability distribution over all letters for the next guess.
        Uses the Viterbi-like logic to combine left-to-current and current-to-right probabilities.
        """
        masked_word = masked_word.lower()
        guessed = set(guessed)
        # Initialize with a tiny probability (smoothing)
        probs = Counter({ch: 1e-6 for ch in self.vocab})
        
        # Iterate through every position in the word
        for i, ch in enumerate(masked_word):
            if ch == "_":
                # Determine left and right context
                left = masked_word[i-1] if i > 0 else "<s>"
                right = masked_word[i+1] if i < len(masked_word) - 1 else "</s>"
                
                # Check every possible letter 'l' for the blank
                for l in self.vocab:
                    # P(l | left) = Count(left -> l) / Count(left)
                    # Use + 1e-6 for smoothing to avoid division by zero
                    p_left = self.bigram_counts[left][l] / (self.unigram_counts[left] + 1e-6)
                    
                    # P(right | l) = Count(l -> right) / Count(l)
                    p_right = self.bigram_counts[l][right] / (self.unigram_counts[l] + 1e-6)
                    
                    # CRITICAL FIX: The joint probability is P(l | left) * P(right | l)
                    # We ADD the joint probability to 'probs[l]' for every blank spot.
                    probs[l] += (p_left * p_right)

        # Mask out letters that have already been guessed
        for g in guessed:
            probs[g] = 0

        # Normalize the probabilities to sum to 1
        total = sum(probs.values())
        if total == 0:
             # Fallback to uniform if no valid probabilities are found
            valid_letters = [l for l in self.vocab if l not in guessed]
            return np.array([1.0 / len(valid_letters) if l in valid_letters else 0.0 for l in self.vocab])

        return np.array([probs[ch] / total for ch in self.vocab])

    def save(self, path):
        """Saves the trained HMM model to a file."""
        joblib.dump(self, path)
        print(f"✅ HMM Model saved to {path}")

    @staticmethod
    def load(path):
        """Loads the HMM model from a file."""
        model = joblib.load(path)
        print(f"✅ HMM Model loaded from {path}")
        return model