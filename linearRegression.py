import numpy
from numpy import array, dot, transpose, random
from numpy.linalg import inv

def linearRegression(data):
	# input - data (x,y)
	# output - w=(x'.x)^-(1).x'.y

	# separate input/output and add bias=1
	x = array([[1]+list(d[:-1]) for d in data])
	y = array([d[-1] for d in data])

	xT = transpose(x)
	xTxInv = inv(dot(xT,x))

	w = dot(dot(xTxInv,xT),y)
	return w


# Generate toy data within model 
noise = lambda: random.random() * .01
initW = array([-3,10,2,9,1])
bias, data = initW[0], initW[1:]
points = [tuple(v) + (dot(data, v) + bias + noise(),) for v in [random.random(4) for _ in range(100)]]

w = linearRegression(points)
print initW
print w
