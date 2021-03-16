#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(model, axis):
	x0, x1 = np.meshgrid(
		np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
		np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1)
	)
	x_new = np.c_[x0.ravel(), x1.ravel()]
	y_pred = model.predict(x_new)
	zz = y_pred.reshape(x0.shape)

	from matplotlib.colors import ListedColormap
	custom_cmap = ListedColormap(['#EF9A9A','#FFF59D','#90CAF9'])
	plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)

def main():
	from sklearn import datasets
	# x, y = datasets.make_moons()
	x, y = datasets.make_moons(noise=0.15, random_state=666)
	# plt.scatter(x[y == 0, 0], x[y == 0, 1])
	# plt.scatter(x[y == 1, 0], x[y == 1, 1])
	# plt.show()

	# 使用多项式特征的SVM
	from sklearn.preprocessing import PolynomialFeatures, StandardScaler
	from sklearn.svm import LinearSVC
	from sklearn.pipeline import Pipeline

	def PolynomialSVC(degree, C=1.0):
		return Pipeline([
			("poly", PolynomialFeatures(degree=degree)),
			("std_scaler", StandardScaler()),
			("linearSVC", LinearSVC(C=C))
		])
	poly_svc = PolynomialSVC(degree=3, C=1.0)
	poly_svc.fit(x, y)

	plot_decision_boundary(poly_svc, axis=[-1.5, 2.5, -1.0, 1.5])
	plt.scatter(x[y == 0, 0], x[y == 0, 1], color='red')
	plt.scatter(x[y == 1, 0], x[y == 1, 1], color='blue')
	plt.show()

	# 使用多项式核函数的SVM
	from sklearn.svm import SVC

	def Polynomial_kernel_SVC(degree, C=1.0):
		return Pipeline([
			("std_scaler", StandardScaler()),
			("kernelSVC", SVC(kernel="poly", degree=degree, C=C))
		])
	
	poly_kernel_svc = Polynomial_kernel_SVC(degree=3, C=1.0)
	poly_kernel_svc.fit(x, y)

	plot_decision_boundary(poly_kernel_svc, axis=[-1.5, 2.5, -1.0, 1.5])
	plt.scatter(x[y == 0, 0], x[y == 0, 1], color='red')
	plt.scatter(x[y == 1, 0], x[y == 1, 1], color='blue')
	plt.show()

if __name__ == "__main__":
	main()
	pass