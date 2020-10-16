from mat4py import loadmat
from sklearn.model_selection import train_test_split
import pickle

data = loadmat("data/data.mat")
features, labels = data['X'], data['y']
for i, label in enumerate(labels):
    labels[i] = label[0] % 10

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25)

filename = "data/train_data"
outfile = open(filename, 'wb')
pickle.dump((features_train, labels_train), outfile)
outfile.close()

filename = "data/test_data"
outfile = open(filename, 'wb')
pickle.dump((features_test, labels_test), outfile)
outfile.close()
