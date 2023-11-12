import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

np.random.seed(54)
heights = np.random.normal(loc=165, scale=10, size=100)  
arm_lengths = 0.7 * heights + 20 + np.random.normal(scale=5, size=100)

heights_train, heights_test, arms_train, arms_test = train_test_split(heights, arm_lengths, test_size=0.2, random_state=54)

linear_model = LinearRegression()
linear_model.fit(heights_train.reshape(-1, 1), arms_train)

tree_model = DecisionTreeRegressor()
tree_model.fit(heights_train.reshape(-1, 1), arms_train)

linear_predictions = linear_model.predict(heights_test.reshape(-1, 1))
tree_predictions = tree_model.predict(heights_test.reshape(-1, 1))

weights = [0.6, 0.4]

ensemble_predictions = weights[0] * linear_predictions + weights[1] * tree_predictions

mse_ensemble = np.mean((ensemble_predictions - arms_test) ** 2)
print(f"Ensemble Model Mean Squared Error: {mse_ensemble}")

plt.scatter(heights_test, arms_test, label='True Data')
plt.plot(heights_test, linear_predictions, label='Linear Model', linestyle='--')
plt.plot(heights_test, ensemble_predictions, label='Ensemble Model', linestyle='--')
plt.xlabel('Height')
plt.ylabel('Arm Length')
plt.legend()
plt.show()
