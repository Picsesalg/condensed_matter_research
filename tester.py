"""
Alice Yang
Department of Physics and Astronomy
The Johns Hopkins University
Baltimore, 21218
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import random
import datetime
import math
import resource

"""
Initialising the randomisations.
"""
now = datetime.datetime.now()
seed = now.hour * 3600 + now.minute * 60 + now.second
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

"""
Iniatialising the variables.
"""

#Number of features on one side of the lattice structure.
L = 30
#Number of inputs.
ni = L * L
#Number of nodes in the hidden layer.
nn = 10
#Number of outputs.
no = 1 

#Number of training samples.
nsap = 20000
#Number of validation samples.
nval = 4000

#Size of mini-batch.
mini = 10
#Number of iterations of mini-batches in each epoch.
epoch = 2000

#Learning rate.
eta = 0.16
#L2 regularisation.
lambda1 = 0.001

#Number of times the hyper-parameters are changed.
nstop = 2
#Number of epochs each time before changing hyper-paramters.
nstep = 5
#Used in generating samples.
itval = L * L / 2

#Counter for how many times hyper-parameters have been updated.
ncount = 0
#Counter for number of epochs.
nepoch = 0

"""
Getting data.
It's faster to feed prepared data into the algorithm, than it is to generate the sample using Python.
samples.csv created by "sample_generate.f".
"""
samples = pd.read_csv("samples.csv")

x = samples.drop(labels=["Label"], axis=1).values
y = samples.Label.values

train_samples = np.random.choice(len(x), 40000, replace=False)
validate_samples = np.array(list(set(range(len(x)))- set(train_samples)))

train_x = x[train_samples]
train_y = y[train_samples]
validate_x = x[validate_samples]
validate_y = y[validate_samples]

"""
Building neural network model.
"""
X = tf.placeholder(tf.float32, [None, ni])
Y = tf.placeholder(tf.float32, [None, no])

weights = {
	'w1': tf.Variable(tf.random_normal(shape=[ni, nn], stddev=1/math.sqrt(ni))),
	'wf': tf.Variable(tf.random_normal(shape=[nn, no], stddev=1/math.sqrt(ni)))
}

biases = {
	'b1': tf.Variable(tf.random_normal(shape=[nn], stddev=1/math.sqrt(ni))),
	'bf': tf.Variable(tf.random_normal(shape=[no], stddev=1/math.sqrt(ni)))
}

def model(X): #{
	z1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
	a1 = tf.sigmoid(z1)
	zf = tf.add(tf.matmul(a1, weights['wf']), biases['bf'])
	return zf
#}

z2 = model(X)

"""
Training model.
"""
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=z2, labels=Y))
reg = tf.nn.l2_loss(weights["w1"]) + tf.nn.l2_loss(weights["wf"])
l2_loss = tf.reduce_mean(loss + reg * lambda1)
optimiser = tf.train.GradientDescentOptimizer(0.1)
train = optimiser.minimize(l2_loss, var_list={weights["w1"], weights["wf"], biases["b1"], biases["bf"]})

"""
Validation model.
"""
a2 = tf.round(tf.sigmoid(z2))
correct = tf.cast(tf.equal(a2, Y), dtype=tf.float32)
accuracy = tf.reduce_mean(correct)

"""
Session starts.
This is when we input the variables into the models above and run the various tensorflow functions.
"""
init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)
while (ncount <= 2): #{
	nepoch += 1
	cost = 0.0
	accu = 0.0
	for i in range (epoch): #{
		batch_index = np.random.choice(len(train_x), size=mini)
		batch_x = train_x[batch_index]
		batch_y = np.matrix(train_y[batch_index]).T
		sess.run(train, feed_dict={X: batch_x, Y: batch_y})
		cost += sess.run(l2_loss, feed_dict={X: batch_x, Y: batch_y})
	#}
	accu = sess.run(accuracy, feed_dict={X: validate_x, Y: np.matrix(validate_y).T})
	print("Epoch = {0}, Accuracy = {1:10.6f}".format(nepoch, accu))
	print("Slowdown # = {0}, Cost = {1: 10.9f}".format(ncount, cost / epoch))
	if (accu >= 0.95 and nepoch//nstep * nstep == nepoch): #{
		ncount += 1
	#}
#}
print("Memory = {0}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print("Training complete.")

end = datetime.datetime.now()
end_seed = end.hour * 3600 + end.minute * 60 + end.second
print("Time = {}".format(end_seed - seed))

temperatures = pd.read_csv("temperatures.csv")
x = temperatures.drop(labels="Label", axis=1).values
y = temperatures.Label.values
for n in range(76): #{
	T = 3.5 - n * 0.025
	temp_range = np.arange(n * 4000, (n + 1) * 4000)
	temp_x = np.take(x, temp_range, axis=0)
	temp_y = np.take(y, temp_range, axis=0)
	temp_y = np.reshape(temp_y, (4000, 1))
	accu = sess.run(accuracy, feed_dict={X: temp_x, Y: temp_y})
	print("T/J = {0}, Ordering Probability = {1: 10.9f}".format(T, accu))
	print("Memory = {0}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
#}

print("Time = {0}".format(end_seed - seed))

