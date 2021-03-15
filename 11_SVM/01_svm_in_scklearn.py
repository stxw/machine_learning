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

def plot_svc_decision_boundary(model, axis):
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

	# 绘制 margin 范围
	w = model.coef_[0]
	b = model.intercept_[0]

	plot_x = np.linspace(axis[0], axis[1], 200)
	up_y = -w[0] / w[1] * plot_x - b / w[1] + 1.0/w[1]
	down_y = -w[0] / w[1] * plot_x - b / w[1] - 1.0/w[1]

	up_index = (up_y >= axis[2]) & (up_y <= axis[3])
	down_index = (down_y >= axis[2]) & (down_y <= axis[3])
	plt.plot(plot_x[up_index], up_y[up_index], color="black")
	plt.plot(plot_x[down_index], down_y[down_index], color="black")

def main():
	# 加载数据
	from sklearn import datasets
	iris = datasets.load_iris()
	x = iris.data
	y = iris.target
	# print(x.shape, y.shape)
	x = x[y < 2, :2]
	y = y[y < 2]

	# 数据可视化
	# plt.scatter(x[y == 0, 0], x[y == 0, 1], color='red')
	# plt.scatter(x[y == 1, 0], x[y == 1, 1], color='blue')
	# plt.show()

	# 数据归一化
	from sklearn.preprocessing import StandardScaler
	standardScaler = StandardScaler()
	standardScaler.fit(x)
	x_standard = standardScaler.transform(x)

	# 训练 hard svm
	from sklearn.svm import LinearSVC
	svc = LinearSVC(C=1e9)
	svc.fit(x_standard, y)

	# 可视化 hard svm 的训练结果
	plot_svc_decision_boundary(svc, axis=[-3, 3, -3, 3])
	plt.scatter(x_standard[y == 0, 0], x_standard[y == 0, 1], color='red')
	plt.scatter(x_standard[y == 1, 0], x_standard[y == 1, 1], color='blue')
	plt.show()

	# 训练 soft svm
	svc2 = LinearSVC(C=0.01) 
	svc2.fit(x_standard, y)

	# 可视化 soft svm 的训练结果
	plot_svc_decision_boundary(svc2, axis=[-3, 3, -3, 3])
	plt.scatter(x_standard[y == 0, 0], x_standard[y == 0, 1], color='red')
	plt.scatter(x_standard[y == 1, 0], x_standard[y == 1, 1], color='blue')
	plt.show()

if __name__ == "__main__":
	main()
	pass