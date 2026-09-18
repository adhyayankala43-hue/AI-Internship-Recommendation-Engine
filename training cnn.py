import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import numpy as np

print("🚀 Loading databases for Deep Learning...")
interactions = pd.read_csv(r"D:\recomm\user_interactions_timeline.csv")
internships = pd.read_csv(r"D:\recomm\internship_database_2.csv")

# 1. Prepare the Data
# We encode the company names into numbers so the Neural Network can understand them
company_encoder = LabelEncoder()
interactions['Company_ID'] = company_encoder.fit_transform(interactions['Company_Name'])

# Normalize the interaction scores between 0 and 1
interactions['Score_Normalized'] = interactions['Interaction_Score'] / 5.0

X = torch.tensor(interactions['Company_ID'].values, dtype=torch.long)
y = torch.tensor(interactions['Score_Normalized'].values, dtype=torch.float32).unsqueeze(1)

num_companies = len(company_encoder.classes_)

# 2. Build the Neural Network Architecture
class RecommendationNet(nn.Module):
    def __init__(self, num_companies, embedding_dim=16):
        super(RecommendationNet, self).__init__()
        # Embedding layer learns a dense vector representation of each company
        self.company_embedding = nn.Embedding(num_companies, embedding_dim)
        
        # Hidden layers to find complex patterns
        self.fc1 = nn.Linear(embedding_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid() # Outputs a strict percentage between 0 and 1

    def forward(self, company_idx):
        embed = self.company_embedding(company_idx)
        x = self.relu(self.fc1(embed))
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.out(x))

model = RecommendationNet(num_companies)

# 3. Train the Model
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("🧠 Training Deep Learning Model...")
epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

# 4. Save the Trained Weights and Encoders
torch.save(model.state_dict(), r"D:\recomm\dl_model.pth")
np.save(r"D:\recomm\company_classes.npy", company_encoder.classes_)

print("✅ Deep Learning Model trained and saved successfully as dl_model.pth!")
