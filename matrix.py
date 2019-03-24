import numpy as np

airMethod = ['2', '3', '1', '4']
energyMethod = ['4', '2', '3', '1']
waterMethod = ['2', '4', '1', '3']
landfillMethod = ['3', '2', '4', '1']
userWant = np.array([])

print("Rank your priorities from 1 to 4:")
airRank = np.array(input("Air Rank (1-4): "))
energyRank = np.array(input("Energy Rank (1-4): "))
waterRank = np.array(input("Water Rank (1-4): "))
landfillRank = np.array(input("Landfill Rank (1-4): "))

userWant = np.append(userWant, airRank)
userWant = np.append(userWant, energyRank)
userWant = np.append(userWant, waterRank)
userWant = np.append(userWant, landfillRank)

airMethodError = np.array(userWant - airRank)
print(airMethodError)
