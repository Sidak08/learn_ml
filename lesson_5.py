#qestions
# why did the batch size change from 32 to 256
# what is the difference between the two input functions
# how many dif models are there
# no epochs here

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

train.head()

train_y = train.pop('Species')
test_y = test.pop('Species')
train.head()

print(train.shape)


def input_fn(features, labels, training=True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)


my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)


classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[30, 10],
    n_classes=3)

#lambda are preety cool
labmdaCool = lambda: print("cool")

classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)
#steps replace ephocs

eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))

#we dont need to use 2 functions cause of mabda


print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
#atleast it is better than vid Test set accuracy: 0.500

def input_fn(features, batch_size=256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as prompted.")
for feature in features:
  valid = True
  while valid:
    val = input(feature + ": ")
    if not val.isdigit(): valid = False

  predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: input_fn(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%)'.format(
        SPECIES[class_id], 100 * probability))

# some things to try
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}

# Please type numeric values as prompted.
# SepalLength: 5.1
# SepalWidth: 3.3
# PetalLength: 1.7
# PetalWidth: 0.5
# Prediction is "Setosa" (70.2%)
# Ok it got it right

# Please type numeric values as prompted.
# SepalLength: 5.9
# SepalWidth: 3.0
# PetalLength: 4.2
# PetalWidth: 1.5
# Prediction is "Virginica" (51.5%)
# Ok two in row but the confidence was low

# Please type numeric values as prompted.
# SepalLength: 6.9
# SepalWidth: 3.1
# PetalLength: 5.4
# PetalWidth: 2.1
# Prediction is "Virginica" (57.2%)
