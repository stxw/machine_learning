import numpy as np

class LinearRegression:

	def __init__(self):
		self.coef_ = None
		self.interception_ = None
		self.theta_ = None

	def fit_normal(self, X_train, Y_train):
		assert X_train.shape[0] == Y_train.shape[0], \
			"the size of X_train must equal to the size of Y_train"
		
		X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
		self.theta_ = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y_train)
		self.interception_ = self.theta_[0]
		self.coef_ = self.theta_[1:]
		return self
	
	def predict(self, X_test):
		X_b = np.hstack([np.ones((len(X_test), 1)), X_test])
		return X_b.dot(self.theta_)
	
	def score(self, Y_test, Y_predict):
		return 1 - np.sum((Y_predict - Y_test) ** 2) / len(Y_test) / np.var(Y_test)