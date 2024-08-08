# qestions
# what happens when you pass reshape value that does not add up
# what does tensorflow eval look like
# what does tensorlfow infer when given invalid value to reshape

# tutarial link https://www.freecodecamp.org/learn/machine-learning-with-python/tensorflow/introduction-to-tensorflow
# collab book link https://colab.research.google.com/drive/1F_EWVKa8rbMXi3_fG0w7AtcscFq7Hi7B#forceEdit=true&sandboxMode=true&scrollTo=Wd85uGI7qyfC

import tensorflow as tf  # now import the tensorflow module
# print(tf.version)  # make sure the version is 2.x

string = tf.Variable("this is a string", tf.string)
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float64)

#me writing it again to help remember

string = tf.Variable("idk", tf.string)
number = tf.Variable(69, tf.int8)
floating = tf.Variable(9.969, tf.float16)


rank1_tensor = tf.Variable(["Test"], tf.string)
rank2_tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string)

# print(tf.rank(rank2_tensor))

# print(rank2_tensor.shape)

tensor1 = tf.ones([1,2,3])  # tf.ones() creates a shape [1,2,3] tensor full of ones
tensor2 = tf.reshape(tensor1, [2,3,1])  # reshape existing data to shape [2,3,1]
tensor3 = tf.reshape(tensor2, [3, -1])

# print(tensor1.shape, tensor2.shape, tensor3.shape)

# print(tensor1)
# print(tensor2)
# print(tensor3)

matrix = [[1,2,3,4,5],
          [6,7,8,9,10],
          [11,12,13,14,15],
          [16,17,18,19,20]]

tensor = tf.Variable(matrix, dtype=tf.int32)
# print(tf.rank(tensor))
# print(tensor.shape)

three = tensor[0,2]  # selects the 3rd element from the 1st row
# print(three)  # -> 3

row1 = tensor[0]  # selects the first row
# print(row1)

column1 = tensor[:, 0]  # selects the first column
# print(column1)

row_2_and_4 = tensor[1::2]  # selects second and fourth row
# print(row2and4) # lol the turorial has an erorr in it
# print(row_2_and_4)

column_1_in_row_2_and_3 = tensor[1:3, 0]
# print(column_1_in_row_2_and_3)

reshapeTen = tf.reshape(tensor, [2, -1])
# print(reshapeTen)

#reshapeBroke = tf.reshape(tensor, [7, 7])
#print(reshapeTen)
# qestion num one answered: it just throws error should have expected that

# reshapeAuto = tf.reshape(tensor, [6, -1])
# print(reshapeAuto)
#qestion number three answer: lol just throws another erorr

# with tf.Session() as sess:
#     tensor.eval()
#     print(tensor.eval())
#     print(tensor)

#qestion number three answer: no clue maybe wrong version but the propety does not exist
