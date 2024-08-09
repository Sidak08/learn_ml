#qestions
# none this time

#notes
# learned about k classifications model but dint implement
# for futher refernce on how it work
# first K amount of centeorids are placed radmoly
# then all points are assigned to the nearest centroid
# then the centroid is moved to the center of the points assigned to it
# then the points are reassigned to the nearest centroid
# and the cycle repeats till the centroid does not move
#
# hidden markov model
# it is preety usless | could be done with recursion
# returns the same answer every time
# it has satates and observations
# the states are the hidden and the observations are the outputs

import tensorflow_probability as tfp
import tensorflow as tf

tfd = tfp.distributions
initial_distribution = tfd.Categorical(probs=[0.2, 0.8])
transition_distribution = tfd.Categorical(probs=[[0.5, 0.5],
                                                 [0.2, 0.8]])
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])

model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)

mean = model.mean()

with tf.compat.v1.Session() as sess:
  print(mean.numpy())
