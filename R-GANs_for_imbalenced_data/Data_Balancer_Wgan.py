import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

class DataBalancerWGAN:
    def __init__(self, latent_dim=10, hidden_dim=16, epochs=5000, batch_size=64, lr=0.0002, lambda_gp=10):
        """
        Initialize the WGAN-GP model parameters for synthetic data generation.
        """
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lambda_gp = lambda_gp
        self.generator = None
        self.critic = None

    class Generator(nn.Module):
        def __init__(self, latent_dim, data_dim, hidden_dim):
            """Define the generator architecture."""
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, data_dim)
            )

        def forward(self, z):
            """Forward pass of the generator."""
            return self.model(z)

    class Critic(nn.Module):
        def __init__(self, data_dim, hidden_dim):
            """Define the critic (discriminator) architecture."""
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(data_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        def forward(self, x):
            """Forward pass of the critic."""
            return self.model(x)

    def gradient_penalty(self, critic, real_samples, fake_samples):
        """
        Compute the gradient penalty for stabilizing WGAN-GP training.
        """
        alpha = torch.rand(real_samples.size(0), 1)
        alpha = alpha.expand_as(real_samples)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        critic_interpolates = critic(interpolates)
        gradients = torch.autograd.grad(outputs=critic_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones_like(critic_interpolates),
                                        create_graph=True, retain_graph=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def fit(self, df, target_col, minority_class=None):
        """
        Apply WGAN-GP to balance the dataset by generating synthetic samples.
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in the dataset.")

        counter = Counter(df[target_col])
        if minority_class is None:
            minority_class = min(counter, key=counter.get)
        
        majority_class = max(counter, key=counter.get)
        minority_count = counter[minority_class]
        majority_count = counter[majority_class]
        
        target_minority_count = int(majority_count * 0.8)
        samples_to_generate = max(0, target_minority_count - minority_count)
        
        if samples_to_generate == 0:
            print("Dataset is already balanced. No synthetic data generated.")
            return df
        
        minority_data = df[df[target_col] == minority_class].drop(columns=[target_col]).values
        data_dim = minority_data.shape[1]

        self.generator = self.Generator(self.latent_dim, data_dim, self.hidden_dim)
        self.critic = self.Critic(data_dim, self.hidden_dim)

        optim_g = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.9))
        optim_c = optim.Adam(self.critic.parameters(), lr=self.lr, betas=(0.5, 0.9))

        for epoch in range(self.epochs):
            z = torch.randn(self.batch_size, self.latent_dim)
            fake_data = self.generator(z)
            real_samples = torch.tensor(minority_data[np.random.randint(0, len(minority_data), self.batch_size)], dtype=torch.float32)

            optim_c.zero_grad()
            real_preds = self.critic(real_samples)
            fake_preds = self.critic(fake_data.detach())
            gp = self.gradient_penalty(self.critic, real_samples, fake_data.detach())
            loss_c = -(real_preds.mean() - fake_preds.mean()) + self.lambda_gp * gp
            loss_c.backward()
            optim_c.step()

            optim_g.zero_grad()
            fake_preds = self.critic(fake_data)
            loss_g = -fake_preds.mean()
            loss_g.backward()
            optim_g.step()

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}: Loss Critic = {loss_c.item():.4f}, Loss Generator = {loss_g.item():.4f}")

        z = torch.randn(samples_to_generate, self.latent_dim)
        synthetic_data = self.generator(z).detach().numpy()
        synthetic_data += np.random.normal(0, 0.05, synthetic_data.shape)  # Adding Gaussian noise
        
        synthetic_df = pd.DataFrame(synthetic_data, columns=[col for col in df.columns if col != target_col])
        synthetic_df[target_col] = minority_class
        
        balanced_df = pd.concat([df, synthetic_df], axis=0).reset_index(drop=True)
        
        print("\n🔹 Class distribution after WGAN-GP generation:", Counter(balanced_df[target_col]))
        torch.save(self.generator.state_dict(), "generator_wgan.pth")
        return balanced_df

if __name__ == "__main__":
    
 # Example of use 
    from sklearn.datasets import make_classification
    
    # Generate an imbalanced dataset
    X, y = make_classification(n_samples=5000, n_features=10, n_classes=2, weights=[0.9, 0.1], random_state=42)
    df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])
    df["Target"] = y
    
    print("🔹 Class distribution before balancing:", Counter(df["Target"]))
    
    # Instantiate and apply DataBalancerWGAN
    balancer = DataBalancerWGAN(epochs=3000, batch_size=64)
    balanced_df = balancer.fit(df, target_col="Target")
    
    print("\n✅ Balanced dataset ready for training!")
