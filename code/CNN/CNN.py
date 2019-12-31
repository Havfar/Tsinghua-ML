import numpy as np
import pandas as pd
from CONV import Conv3x3
from MaxPool import MaxPool2
from Softmax import Softmax

class CNN():
  def __init__(self):
    self.conv = Conv3x3(8)  # 28x28x1 -> 26x26x8
    self.pool = MaxPool2()  # 26x26x8 -> 13x13x8
    self.softmax = Softmax(13 * 13 * 8, 10)  # 13x13x8 -> 10



  def forward(self, image, label):
    out = self.conv.forward((image / 255) - 0.5)
    out = self.pool.forward(out)
    out = self.softmax.forward(out)
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc

  def predict(self, test_images):
    predictions = []
    for img in test_images:
      img = img.reshape(28, 28)
      out = self.conv.forward((img / 255) - 0.5)
      out = self.pool.forward(out)
      out = self.softmax.forward(out)
      predictions.append(np.argmax(out))
    return predictions

  def train(self, im, label, lr=.005):
    out, loss, acc = self.forward(im, label)

    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    gradient = self.softmax.backprop(gradient, lr)
    gradient = self.pool.backprop(gradient)
    self.conv.backprop(gradient, lr)

    return loss, acc

  def run(self, train_images, train_labels, epochs ):
    for epoch in range(epochs):
      print('--- Epoch %d ---' % (epoch + 1))

      permutation = np.random.permutation(len(train_images))
      train_images = train_images[permutation]
      train_labels = train_labels[permutation]
      loss = 0
      num_correct = 0
      for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if i % 100 == 99:
          print(
            '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
            (i + 1, loss / 100, num_correct)
          )
          loss = 0
          num_correct = 0
        l, acc = self.train(im, label)
        loss += l
        num_correct += acc


if __name__ == "__main__":
  print("LOADING DATA")
  train_data = pd.read_csv("../../input/train.csv").values
  test_data = pd.read_csv("../../input/test.csv").values
  train_images = train_data[0:, 1:]
  train_labels = train_data[0:, 0]
  test_images = test_data

  train_images = train_images.reshape(train_images.shape[0], 28, 28)
  test_images = test_images.reshape(test_images.shape[0], 28, 28)

  cnn = CNN()
  cnn.run(train_images=train_images, train_labels=train_labels, epochs=5)
  predtictY = cnn.predict(test_data)

  # Create submission file
  df_sub = pd.DataFrame(list(range(1, len(test_data) + 1)))
  df_sub.columns = ["ImageID"]
  df_sub["Label"] = predtictY
  df_sub.to_csv("../../output/havlearn_cnn.csv", index=False)
