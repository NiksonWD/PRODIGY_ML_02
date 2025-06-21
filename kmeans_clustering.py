
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = {
    'CustomerID': range(1, 201),
    'Annual_Income_Rs': np.random.randint(250000, 1500000, 200),
    'Spending_Score': np.random.randint(1, 100, 200)
}
df = pd.DataFrame(data)

X = df[['Annual_Income_Rs', 'Spending_Score']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

k = 5
kmeans = KMeans(n_clusters=k, random_state=0)
df['Cluster'] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Annual_Income_Rs', y='Spending_Score', hue='Cluster', palette='Set1')
plt.title('Customer Segments')
plt.xlabel('Annual Income (Rs)')
plt.ylabel('Spending Score')
plt.legend()
plt.grid(True)
plt.show()
