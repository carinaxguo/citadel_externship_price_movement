"""
Project Team: Carina Guo & Marija Vukic
Our objective is to model the price movement of a derivative asset, conduct a linear regression to determine the relative importance of each underlying asset on the resulting price, then demonstrate that this ordering indeed is valid.

ASSUMPTIONS:
- Our derivative is based on three underlying assets: A, B, and C
- The derivative's price movement is a weighted sum of the movements of A, B, and C, whose weights are in the range -10 to 10 (generated randomly)
- The specific weights are unknown/unnecessary information for the strategies to execute. The only required information is the relative ordering of the assets' importance, and whether the derivative is directly or inversely correlated to the underlying assets.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


class Change_Calc:

  def __init__(self):
    self.coeffs = []
    self.data = []
    self.order = []

  def createFunc(self):
    coeff_a = np.random.randint(-10, 10)
    coeff_b = np.random.randint(-10, 10)
    coeff_c = np.random.randint(-10, 10)
    self.coeffs = [coeff_a, coeff_b, coeff_c]

  def createData(self):
    self.data = []
    for i in range(1000):
      change_a = round(np.random.uniform(-1, 1), 2)
      change_b = round(np.random.uniform(-1, 1), 2)
      change_c = round(np.random.uniform(-1, 1), 2)
      change_asset = round(
        self.coeffs[0] * change_a + self.coeffs[1] * change_b +
        self.coeffs[2] * change_c, 2)
      self.data.append([change_a, change_b, change_c, change_asset])

  def getWeighting(self):
    df = pd.DataFrame(self.data, columns=["A", "B", "C", "Asset"])
    x = df.drop("Asset", axis=1)
    y = df["Asset"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    weights = LinearRegression().fit(x_train, y_train).coef_
    input_to_weight = [("A", weights[0]), ("B", weights[1]), ("C", weights[2])]
    input_to_weight.sort(key=lambda x: abs(x[1]), reverse=True)
    self.order = [pair[0] for pair in input_to_weight]
    return self.order


def strat(coeffs, data, order):
  sums = [0, 0, 0]
  for line in data:
    for i in range(3):
      if line[i] > 0 and coeffs[i] > 0 or line[i] < 0 and coeffs[i] < 0:
        sums[i] += line[3]
      else:
        sums[i] -= line[3]
  return sums


def printSums(sums):
  print(f"Using Asset A for cues: ${round(sums[0],2)}")
  print(f"Using Asset B for cues: ${round(sums[1],2)}")
  print(f"Using Asset C for cues: ${round(sums[2],2)}")
  print()


#RUN 1
test = Change_Calc()
test.createFunc()
test.createData()
test.getWeighting()

print(test.coeffs)
print()

for line in test.data:
  print(line)
print()

print(test.order)
print()

sums = strat(test.coeffs, test.data, test.order)
printSums(sums)

#RUN 2
print("Resetting dataset...")
test.createData()
sums = strat(test.coeffs, test.data, test.order)
printSums(sums)

#100 iterations
y_graph = []
x_graph = range(100)
for i in range(100):
  test.createData()
  sums = strat(test.coeffs, test.data, test.order)
  y_graph.append(max(sums))

average = round(np.average(y_graph),2)
print(f'Average max payoff for 100 sample datasets: ${average}')

plt.plot(x_graph, y_graph, 'o')
plt.ylim(0, average+3000)
plt.show()
