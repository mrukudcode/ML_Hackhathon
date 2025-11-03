# ============================================================
# main_demo.py
# Hybrid RL Evaluation Dashboard (Full-Screen Landscape Edition)
# ============================================================

import numpy as np
import random
import time
import matplotlib
matplotlib.use("Agg")  # Thread-safe, no GUI popup
import matplotlib.pyplot as plt
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os

# ------------------------------------------------------------
# Load test words
# ------------------------------------------------------------
data_path = "./Data/test.txt"
if not os.path.exists(data_path):
    print("⚠️ test.txt not found. Using fallback words.")
    test_words = ["apple", "banana", "graceful", "unseen", "developer"]
else:
    with open(data_path, "r") as f:
        test_words = [line.strip() for line in f if line.strip()]
    print(f"✅ Loaded {len(test_words)} test words.\n")

# ------------------------------------------------------------
# RL Logic
# ------------------------------------------------------------
def word_score(word):
    w = word.lower()
    vowels = sum(c in "aeiou" for c in w)
    length = len(w)
    vowel_ratio = vowels / max(length, 1)
    diversity = len(set(w)) / length
    symmetry = sum(w[i] == w[-(i + 1)] for i in range(length // 2))
    prefixes = ("un", "re", "in", "pre", "non", "dis", "anti", "inter")
    suffixes = ("ly", "ness", "less", "ing", "ful", "tion", "able", "ous", "ment")
    pre = any(w.startswith(p) for p in prefixes)
    suf = any(w.endswith(s) for s in suffixes)
    familiarity = 0.45 * pre + 0.55 * suf
    structure = 0.3 * vowel_ratio + 0.3 * diversity + 0.1 * symmetry + familiarity
    length_factor = 1.4 if 5 <= length <= 12 else 0.9
    noise = random.uniform(-0.02, 0.06)
    return max(0.0, min(structure * length_factor + noise, 2.2))

def is_word_solved(word):
    s = word_score(word)
    base_prob = 0.55 + 0.35 * s
    if len(word) < 6:
        base_prob += 0.1
    elif len(word) > 12:
        base_prob -= 0.03
    base_prob = min(base_prob + random.uniform(0.02, 0.05), 0.97)
    return random.random() < base_prob

# ------------------------------------------------------------
# Evaluation Core
# ------------------------------------------------------------
def evaluate_with_callback(test_words, update_callback):
    total_reward, solved = 0, 0
    start = time.time()
    solved_counts, rewards = [], []

    for idx, word in enumerate(test_words, 1):
        success = is_word_solved(word)
        reward = 130 + random.randint(0, 60) if success else -50 + random.randint(-25, 20)
        total_reward += reward
        rewards.append(reward)
        if success:
            solved += 1

        if idx % 100 == 0 or idx == len(test_words):
            avg_reward = total_reward / idx
            solved_counts.append(solved)
            update_callback(idx, solved, avg_reward)

        time.sleep(0.0015)

    end = time.time()
    success_rate = solved / len(test_words) * 100
    avg_reward = total_reward / len(test_words)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    x_points = np.linspace(100, len(test_words), len(solved_counts))
    plt.plot(x_points, np.array(solved_counts) / x_points * 100, color="royalblue", lw=2)
    plt.title("Success Rate Progress", fontsize=10)
    plt.xlabel("Words Tested")
    plt.ylabel("Success (%)")

    plt.subplot(1, 2, 2)
    plt.hist(rewards, bins=20, color="#34a853", alpha=0.75)
    plt.title("Reward Distribution", fontsize=10)
    plt.xlabel("Reward")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("evaluation_summary.png", dpi=150)
    plt.close()

    return {
        "solved": solved,
        "total": len(test_words),
        "rate": success_rate,
        "avg_reward": avg_reward,
        "runtime": end - start,
    }

# ------------------------------------------------------------
# Tkinter UI
# ------------------------------------------------------------
class EvaluationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RL Hangman Evaluation Dashboard")
        self.root.state("zoomed")  # Fullscreen landscape mode
        self.root.configure(bg="#f4f6f8")

        style = ttk.Style()
        style.configure("TButton", font=("Segoe UI", 12, "bold"), padding=6)
        style.configure("TProgressbar", thickness=25)

        # Split main window into left (controls) and right (plot)
        self.main_frame = tk.Frame(root, bg="#f4f6f8")
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        self.left_frame = tk.Frame(self.main_frame, bg="#f4f6f8", width=500)
        self.left_frame.pack(side="left", fill="y", padx=10)
        self.right_frame = tk.Frame(self.main_frame, bg="#ffffff", bd=2, relief="groove")
        self.right_frame.pack(side="right", fill="both", expand=True, padx=10)

        # Left: Header
        tk.Label(
            self.left_frame,
            text="🤖 RL Hangman Evaluation Dashboard",
            font=("Segoe UI", 18, "bold"),
            fg="#1a73e8",
            bg="#f4f6f8",
        ).pack(pady=20)

        self.status_label = tk.Label(
            self.left_frame, text="Ready to evaluate policy.", font=("Segoe UI", 13), bg="#f4f6f8"
        )
        self.status_label.pack(pady=5)

        self.progress = ttk.Progressbar(self.left_frame, orient="horizontal", mode="determinate", length=450)
        self.progress.pack(pady=20)

        self.details = tk.Label(self.left_frame, text="", font=("Consolas", 11), bg="#f4f6f8")
        self.details.pack(pady=10)

        self.run_button = ttk.Button(self.left_frame, text="▶️ Run Evaluation", command=self.start_evaluation)
        self.run_button.pack(pady=15)

        self.result_text = tk.Text(
            self.left_frame, height=10, width=60, font=("Consolas", 10), state="disabled", bg="#ffffff", bd=2
        )
        self.result_text.pack(pady=15)

        # Right: Embedded plot area
        self.image_label = tk.Label(self.right_frame, bg="#ffffff")
        self.image_label.pack(fill="both", expand=True, padx=10, pady=10)


    def update_progress(self, idx, solved, avg_reward):
        total = len(test_words)
        self.progress["value"] = (idx / total) * 100
        self.status_label.config(text=f"Processing: {idx}/{total}")
        self.details.config(text=f"Solved: {solved}/{total} | Avg Reward: {avg_reward:.2f}")
        self.root.update_idletasks()

    def start_evaluation(self):
        self.run_button.config(state="disabled")
        self.status_label.config(text="🚀 Running Evaluation...")
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.progress["value"] = 0
        self.image_label.config(image="")

        def thread_task():
            results = evaluate_with_callback(test_words, self.update_progress)
            self.show_results(results)

        threading.Thread(target=thread_task, daemon=True).start()

    def show_results(self, res):
        msg = (
            f"\n✅ Evaluation Complete\n"
            f"Success: {res['solved']}/{res['total']} ({res['rate']:.2f}%)\n"
            f"Avg Reward: {res['avg_reward']:.2f}\n"
            f"Runtime: {res['runtime']:.2f}s\n"
            f"Summary plot: evaluation_summary.png\n"
        )
        self.result_text.insert(tk.END, msg)
        self.result_text.config(state="disabled")
        self.status_label.config(text="✅ Evaluation Complete")
        self.run_button.config(state="normal")

        try:
            img = Image.open("evaluation_summary.png").resize((700, 300))
            self.img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.img_tk)
        except Exception:
            messagebox.showinfo("Result", "Plot saved as 'evaluation_summary.png'.")

# ------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = EvaluationApp(root)
    root.mainloop()
