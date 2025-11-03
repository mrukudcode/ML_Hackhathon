class HangmanEnv:
    def __init__(self, word, max_wrong=6):
        self.max_wrong = max_wrong
        self.reset(word)

    def get_masked_word(self):
        return ''.join([ch if ch in self.guessed else '_' for ch in self.word])

    def step(self, letter):
        letter = letter.lower()
        reward = 0
        done = False

        if letter in self.guessed:
            reward = -2
        elif letter in self.word:
            self.guessed.add(letter)
            reward = 10
        else:
            self.guessed.add(letter)
            self.wrong += 1
            reward = -5

        if all(ch in self.guessed for ch in self.word):
            done = True
            reward += 50
        elif self.wrong >= self.max_wrong:
            done = True
            reward -= 50

        return self.get_masked_word(), reward, done

    def reset(self, word):
        self.word = str(word).strip().lower()
        self.guessed = set()
        self.wrong = 0
        return self.get_masked_word()


# --------- Interactive runner (manual words; no datasets) ----------
DEFAULT_ORDER = "etaoinshrdlcumwfgypbvkjxqz"  # used for auto mode

def play_manual(word, max_wrong=6):
    env = HangmanEnv(word, max_wrong=max_wrong)
    total_reward = 0
    print(f"\nNew game started. Word length: {len(word)}")
    print("State:", env.get_masked_word())

    while True:
        guess = input("Enter a letter (or 'quit' to stop this word): ").strip().lower()
        if guess == "quit":
            print("Stopping this word.\n")
            break
        if len(guess) != 1 or not guess.isalpha():
            print("Please enter a single alphabetic character.\n")
            continue

        state, reward, done = env.step(guess)
        total_reward += reward
        print(f"Guess '{guess}' → {state} | Reward={reward} | Wrong={env.wrong}/{env.max_wrong}")

        if done:
            if state == env.word:
                print(f"Game over ✅ You solved it! Word: '{env.word}' | Total Reward={total_reward}\n")
            else:
                print(f"Game over ❌ You lost. Word was: '{env.word}' | Total Reward={total_reward}\n")
            break

def play_auto(word, max_wrong=6, order=DEFAULT_ORDER):
    env = HangmanEnv(word, max_wrong=max_wrong)
    total_reward = 0
    tried = set()
    print(f"\nAuto-test for word '{word}' (len={len(word)})")
    print("Start:", env.get_masked_word())

    for ch in order:
        if ch in tried:
            continue
        tried.add(ch)
        state, reward, done = env.step(ch)
        total_reward += reward
        print(f"  guess '{ch}' → {state} | r={reward} | wrong={env.wrong}/{env.max_wrong}")
        if done:
            if state == env.word:
                print(f"Result: WIN ✅ | steps={len(tried)} | total_reward={total_reward}\n")
            else:
                print(f"Result: LOSS ❌ | steps={len(tried)} | total_reward={total_reward}\n")
            return
    print(f"Stopped (ran out of letters). Final state: {env.get_masked_word()} | reward={total_reward}\n")


if __name__ == "__main__":
    print("=== Hangman (Manual Words) ===")
    print("Enter any word to test. Press ENTER with no input to exit.")
    while True:
        w = input("\nWord to test (blank to quit): ").strip()
        if not w:
            print("Bye!")
            break
        if not w.isalpha():
            print("Please enter letters only (no spaces/numbers).")
            continue

        mode = input("Mode? [m]anual guesses / [a]uto test: ").strip().lower()
        if mode not in ("m", "a"):
            print("Invalid choice. Defaulting to manual.")
            mode = "m"

        try:
            mw = input("Max wrong guesses (default 6): ").strip()
            max_wrong = int(mw) if mw else 6
        except ValueError:
            print("Invalid number. Using default 6.")
            max_wrong = 6

        if mode == "m":
            play_manual(w, max_wrong=max_wrong)
        else:
            play_auto(w, max_wrong=max_wrong)
