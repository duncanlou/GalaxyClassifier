import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/training_notes")

training_loss = list(df['Training_epoch_loss'])
validation_loss = list(df[' Validation_epoch_loss'])
training_acc = list(df[' Training_epoch_accuracy'])
validation_acc = list(df[' Validation_epoch_accuracy'])

epochs = range(25)
plt.plot(epochs, training_loss)
plt.show()
