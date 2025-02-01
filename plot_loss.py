import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("training_loss.csv")

ax = df.plot(
  x="epoch", y="loss", kind="line", logy=True, figsize=(12,6), grid=True
)

ax.set_title("Training Loss over Epochs")
ax.set_xlabel("Epoch", fontsize=14)
ax.set_ylabel("Loss", fontsize=14)

plt.show()