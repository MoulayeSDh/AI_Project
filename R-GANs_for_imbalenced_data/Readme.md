# DataBalancerWGAN - Balance Your Dataset with GANs

![GitHub Repo stars](https://img.shields.io/github/stars/moulaye-sidi-dahi/DataBalancerWGAN?style=social)
![GitHub forks](https://img.shields.io/github/forks/moulaye-sidi-dahi/DataBalancerWGAN?style=social)
![GitHub license](https://img.shields.io/badge/license-NY--License-blue)

## 📌 Overview
`DataBalancerWGAN` is a Python class that uses **Wasserstein GAN with Gradient Penalty (WGAN-GP)** to generate synthetic samples for the minority class in imbalanced datasets. Unlike traditional oversampling methods like **SMOTE**, this method **learns the real data distribution** and generates more diverse and realistic samples.

## 🚀 Features
- **Automatic Minority Class Detection** (if not provided).
- **Stable Training with WGAN-GP** to avoid mode collapse.
- **Noise Injection for More Diversity** in generated samples.
- **Fast Integration** into any machine learning pipeline.
- **Pre-Trained Generator Saving & Loading**.
- **Feature Distribution Visualization**.

## 🔧 Installation
This implementation requires Python **3.8+** and the following dependencies:

```
pip install numpy pandas torch matplotlib seaborn scikit-learn
```

## 📖 Usage Example
### 1️⃣ Import & Generate an Imbalanced Dataset
```python
from sklearn.datasets import make_classification
import pandas as pd
from collections import Counter
from data_balancer_wgan import DataBalancerWGAN

# Create an imbalanced dataset
X, y = make_classification(n_samples=5000, n_features=10, n_classes=2, weights=[0.9, 0.1], random_state=42)
df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])
df["Target"] = y

print("🔹 Class distribution before balancing:", Counter(df["Target"]))
```

### 2️⃣ Apply WGAN-GP to Balance Data
```
# Instantiate and apply DataBalancerWGAN
balancer = DataBalancerWGAN(epochs=3000, batch_size=64)
balanced_df = balancer.fit(df, target_col="Target")

print("\n✅ Balanced dataset ready for training!")
```

### 3️⃣ Visualize Feature Distribution
```
balancer.evaluate_distribution(balanced_df, target_col="Target")
```

## 🎯 Key Advantages Over Traditional Methods
| Method | Limitation |
|---------|------------|
| **SMOTE** | Only interpolates existing points, does not learn real distribution. |
| **ADASYN** | Can introduce unnecessary noise. |
| **WGAN-GP** | Generates more diverse & realistic samples, improves generalization. |

## 📂 File Structure
```
📦 data_balancer_wgan
 ┣ 📜 data_balancer_wgan.py # Main class implementation
 ┣ 📜 README.md # Documentation
 ┣ 📜 generator_wgan.pth # Saved generator model (after training)(only if you use the model)
```

## 📌 Future Improvements
- Adding **WGAN-GP with spectral normalization** for even better training stability.
- Implementing **conditional GANs (cGANs)** for multi-class generation.
- Supporting **image data augmentation** for computer vision tasks.

## 📜 License
This project is licensed under **NY-License**. Any **use, modification, or fork must include the author's name (Moulaye Sidi Dahi)** in all distributions.

## 👨‍💻 Author
Developed by **Moulaye Sidi Dahi** | Machine Learning Engineer | Data Scientist | Kaggle Master

🚀 Connect on **[LinkedIn](https://www.linkedin.com/in/moulaye-sidi-dahi/)** for more insights on AI & Data Science!



