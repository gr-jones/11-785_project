import matplotlib
import matplotlib.pyplot as plt
import pandas as pd 

df_ga = pd.read_csv("GradApprox.csv")
df_eve = pd.read_csv("EventProp_og_code.csv")


# Extract data
train_loss_ga = df_ga['train_loss']
train_acc_ga = df_ga['train_accuracy']
test_acc_ga = df_ga['test_accuracy']

train_loss_eve = df_eve['train_loss']
train_acc_eve = df_eve['train_accuracy']
test_acc_eve = df_eve['test_accuracy']


# Plot data 
x = range(len(train_loss_ga))

# Plot EventProp and Grad Approx. training loss
plt.plot(x, train_loss_eve, '-', label='EventProp')
plt.plot(x, train_loss_ga, '-', label='Gradient Approximation')

plt.legend(loc='best')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")

plt.savefig('training_loss_updated_data_encode.png')
plt.show()
plt.close()

# Plot EventProp and Grad Approx. training and validation accuracy
plt.plot(x, train_acc_eve, '-', label='EventProp - Training Accuracy')
plt.plot(x, test_acc_eve, '-', label='EventProp - Validation Accuracy')

plt.plot(x, train_acc_ga, '-', label='Gradient Approximation - Training Accuracy')
plt.plot(x, test_acc_ga, '-', label='Gradient Approximation - Validation Accuracy')

plt.legend(loc='best')

plt.xlabel("Epoch")
plt.ylabel("Classification Accuracy %")
plt.title("Classification Performance of Backprop Algorithms")

plt.savefig('backprop_algos_performance_updated_data_encode.png')
plt.show()
plt.close()