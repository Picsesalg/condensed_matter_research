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
print("Training complete.")

end = datetime.datetime.now()
end_seed = end.hour * 3600 + end.minute * 60 + end.second
print("Time = {}".format(end_seed - seed))

x1 = np.ones((L, L))
print("Calculating phase diagram.")
x1 = np.ones((L, L))
if (random.random() > 0.5): x1 *= 1
for n in range(101): #{
	T = 3.5 - n * 0.025
	for m in range(nval): #{
		for k in range(itval): #{
			i = int(random.random() * L)
			j = int(random.random() * L)
			if (i >= L): i -= 1
			if (j >= L): j -= 1
			ip1 = i + 1
			if (ip1 >= L): ip1 -= L
			im1 = i - 1
			if (im1 < 0): im1 += L
			jp1 = j + 1
			if (jp1 >= L): jp1 -= L
			jm1 = j - 1
			if (jm1 < 0): jm1 += L
			engy = -x1[i][j] * (x1[ip1][j] + x1[im1][j] + x1[i][jp1] + x1[i][jm1])
			if (engy > 0 or random.random() < math.exp(engy * 2 / T)): x1[i][j] *= -1
		#}
		x2 = np.reshape(x1, (1, ni))
		if (m == 0): #{
			T_x = x2
		#}
		else: #{
			T_x = np.vstack((T_x, x2))
		#}
	#}
	T_y = np.full((nval, 1), 1.0)
	accu = sess.run(accuracy, feed_dict={X: T_x, Y: T_y})
	print("T/J = {0}, Ordering Probability = {1: 10.9f}".format(T, accu))
#}


end = datetime.datetime.now()
end_seed = end.hour * 3600 + end.minute * 60 + end.second

print("Time = {0}".format(end_seed - seed))

