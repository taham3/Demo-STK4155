import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


n_x=100   # number of points
m=5        # degree of polynomial

# sort the random values, else your fit will go crazy
x = np.sort(np.random.uniform(0, 1, n_x))
y = np.sort(np.random.uniform(0, 1, n_x))

# use the meshgrid functionality, very useful
x, y = np.meshgrid(x,y)

def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4

z = FrankeFunction(x, y)

#Transform from matrices to vectors
x_1=np.ravel(x)
y_1=np.ravel(y)
n=int(len(x_1))
z_1=np.ravel(z)+ np.random.randn(n)
z_true = np.ravel(z)



# finally create the design matrix
# This design was taken from piazza, we strugled for a long time trying to do it ourself.

def create_X(x, y, n = 5):
	"""
	Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
	Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
	"""
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = x**(i-k) * y**k

	return X

X= create_X(x_1,y_1,n=m)

beta = np.dot((np.dot(np.linalg.inv(np.dot(X.T,X)),X.T)),z_1)

ztilde = np.dot(X,beta)


def MSEfunc (z_1,ztilde):
	MSESUM = 0
	for i in range (n-1):
		MSE2 = (z_1[i]-ztilde[i])**2 
		MSESUM += MSE2
	return MSESUM

MSESUM = (MSEfunc(z_true,ztilde))/n
MSE = mean_squared_error(z_true, ztilde)
print "The sci-kit learn value of MSE is %.10f" %(MSE)
print "The calculated value of MSE is %.10f" %(MSESUM)

mean_z = np.mean(z_true)

def r2score(z_1,ztilde,mean_z):
	sum1 = 0
	sum2 = 0
	for i in range (n-1):
		term1 = (z_1[i] - ztilde[i])**2
		term2 = (z_1[i] - mean_z)**2
		sum1 += term1
		sum2 += term2
	return sum1/sum2
r2analytic = 1 - r2score(z_true,ztilde,mean_z)
r2 = r2_score(z_true,ztilde)

print "The sci-kit learn value of R^2 is %.10f" %(r2)
print "The calculated value of R^2 is %.10f" %(r2analytic)



