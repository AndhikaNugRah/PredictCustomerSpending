#This model created to illustrates 100 customers in a shop, and their shopping habits.

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)

#The x axis represents the number of minutes before making a purchase.
x=np.random.normal(3,1,100)

#The y axis represents the amount of money spent on the purchase.
y=np.random.normal(150,40,100) / x

plt.xlabel ('Number of Minutes')
plt.ylabel ('Amount Spent')
plt.title ('Correlations of Times and Amount Spent (random data)')

plt.scatter(x,y)
plt.show()

#Train the data 80 % and using 20% for test
data_train_x=x[:80]
data_train_y=y[:80]

plt.xlabel ('Number of Minutes')
plt.ylabel ('Amount Spent')
plt.title ('Correlations of Times and Amount Spent (data training)')

plt.scatter(data_train_x,data_train_y)
plt.show()

plt.xlabel ('Number of Minutes')
plt.ylabel ('Amount Spent')
plt.title ('Correlations of Times and Amount Spent (data test)')

data_test_x=x[:20]
data_test_y=y[:20]

plt.scatter(data_test_x,data_test_y)
plt.show()

model_used=np.poly1d(np.polyfit(data_train_x,data_train_y,4))
lined=np.linspace(0,6,100)

plt.xlabel ('Number of Minutes')
plt.ylabel ('Amount Spent')
plt.title ('Correlations of Times and Amount Spent (Fit the Data Set)')

plt.scatter(data_train_x,data_train_y)
plt.plot(lined,model_used(lined))
plt.show()

#find out r2 score to measures the relationship between the x axis and the y axis
from sklearn.metrics import r2_score
print (r2_score(data_train_y,model_used(data_train_x)))

def predict_amount():
    while True:
        try:
            X1 = int(input("Input Time Spent: "))
            break
        except ValueError:
            print("Invalid input. Please enter an integer value.")

    Model_Pred = model_used(X1)
    print(f"if customer spend time {X1} Minutes its possible to spend money : {Model_Pred} Dollars")
    cont = input("Do you want to continue? (yes/no): ")
    if cont.lower() == "yes":
        predict_amount()  # recursive call to go back to the beginning
    else:
        print("Goodbye, thanks for using our service!  - Dhika")

predict_amount()  # initial call to start the loop
#Because the data is centered around a mean of 3 and the sample size is only 100, 
# #this model will likely work best for values up to 6 minutes, as shown in the graph. 
# If we want to make the model more generalizable to a wider range of values, we can adjust the random value generation as needed.