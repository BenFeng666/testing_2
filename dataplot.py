import json
import matplotlib.pyplot as plt
import os

# ===== Paths =====
base_dir = "/content/testing_2/output"
correct_path = f"{base_dir}/correct.json"
wrong_path = f"{base_dir}/wrong.json"
save_path = f"{base_dir}/fine_tuned_plot.png"

# ===== Load data =====
with open(correct_path, "r") as f:
    correct = json.load(f)
with open(wrong_path, "r") as f:
    wrong = json.load(f)

# ===== Extract x (confidence) and y (true score) =====
x_correct = [d["confidence"] for d in correct]
y_correct = [d["ground_truth"] for d in correct]

x_wrong = [d["confidence"] for d in wrong]
y_wrong = [d["ground_truth"] for d in wrong]

# ===== Create scatter plot =====
plt.figure(figsize=(9, 6))
plt.scatter(x_correct, y_correct, color='royalblue', label='Correct', s=45, alpha=0.8)
plt.scatter(x_wrong, y_wrong, color='deeppink', marker='x', label='Wrong', s=70, alpha=0.8)

plt.title("Fine Tuned", fontsize=14)
plt.xlabel("Confidence Score", fontsize=12)
plt.xlim(1, 10)
plt.ylabel("Transfection Efficiency (True Score)", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)

# ===== Save and show =====
os.makedirs(base_dir, exist_ok=True)
plt.tight_layout()
plt.savefig(save_path, dpi=300)
print(f"✅ Figure saved to: {save_path}")

# Optional: show inline if running inside Colab
plt.show()

