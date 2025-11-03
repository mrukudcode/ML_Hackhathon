# hangman_manual.py

import random
import string

class HangmanEnv:
    """
    A simple Hangman game environment for reinforcement learning.
    """
    def __init__(self, word, max_wrong=6):
        self.word = word.lower()
        self.max_wrong = max_wrong
        self.guessed_letters = set()
        self.wrong = 0
        self.done = False

    def get_masked_word(self):
        """Returns the current state of the word (e.g., 'a__l_')."""
        masked = ""
        for char in self.word:
            if char in self.guessed_letters:
                masked += char
            else:
                masked += "_"
        return masked

    def step(self, action):
        """
        Takes a guess (action) and returns the next state, reward, and done status.

        Args:
            action (str): The letter guess.

        Returns:
            tuple: (masked_word, reward, done)
        """
        if self.done:
            return self.get_masked_word(), 0.0, True

        action = action.lower()
        
        # 1. Handle Invalid/Repeat Guesses (Handled by the agent reward scheme, but tracked here)
        if action in self.guessed_letters:
            # Environment treats repeated guess as no-op but the Agent penalizes it
            return self.get_masked_word(), 0.0, False 

        self.guessed_letters.add(action)

        # 2. Update state
        reward = 0.0
        
        if action in self.word:
            # Correct guess
            pass 
        else:
            # Wrong guess
            self.wrong += 1

        masked_word = self.get_masked_word()

        # 3. Check termination conditions
        win = ("_" not in masked_word)
        loss = (self.wrong >= self.max_wrong)
        self.done = win or loss

        # Reward shaping is handled in the RL agent for flexibility
        return masked_word, reward, self.done