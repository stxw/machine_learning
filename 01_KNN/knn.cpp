/* K-近邻算法 */
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <map>

using namespace std;

template <typename x_type, typename y_type>
class KNN_classifier
{
	private:
		int _k;
		vector<vector<x_type> > *_x_train;
		vector<y_type> *_y_train;
	
	public:
		KNN_classifier(int k)
		{
			_k = k;
		}

		void fit(vector<vector<x_type> > *x_train, vector<y_type> *y_train)
		{
			_x_train = x_train;
			_y_train = y_train;
		}

		y_type predict(vector<x_type> *x_test)
		{
			int i, j, max_t;
			y_type predict;
			vector<int> index(_x_train->size());
			vector<double> dist(_x_train->size());
			map<y_type, int> mp;

			for(i = 0; i < dist.size(); i++)
			{
				dist[i] = 0.0;
				for(j = 0; j < x_test->size(); j++)
				{
					dist[i] += pow(_x_train->at(i)[j] - x_test->at(j), 2);
				}
				index[i] = i;
			}
			sort(index.begin(), index.end(), [&dist](int a, int b)
			{
				return dist[a] < dist[b];
			});
			for(i = 0; i < _k; i++)
			{
				mp[ _y_train->at( index[i] ) ]++;
			}

			max_t = 0;
			typename map<y_type, int>::iterator iter;
			for(iter = mp.begin(); iter != mp.end(); iter++)
			{
				if(iter->second > max_t)
				{
					max_t = iter->second;
					predict = iter->first;
				}
			}
			return predict;
		}

		void predict(vector<vector<x_type> > *x_test, vector<y_type> *y_test)
		{
			int i = 0;
			for(i = 0; i < (*x_test).size(); i++)
			{
				y_test->at(i) = predict( &( x_test->at(i) ) );
			}
		}
};

int main()
{
	int i, j, n, m;
	FILE *fp_iris;
	KNN_classifier<double, int> knn(3);

	fp_iris = fopen("./knn/iris.txt", "r");
	fscanf(fp_iris, "%d%d", &m, &n);
	vector<vector<double> > x(m, vector<double> (n));
	vector<int> y(m);
	for(i = 0; i < m; i++)
	{
		for(j = 0; j < n; j++)
		{
			fscanf(fp_iris, "%lf", &x[i][j]);
		}
		fscanf(fp_iris, "%d", &y[i]);
	}
	fclose(fp_iris);
	knn.fit(&x, &y);

	vector<double> x_test(n);
	int predict;
	for(i = 0; i < n; i++)
	{
		scanf("%lf", &x_test[i]);
	}
	predict = knn.predict(&x_test);
	printf("%d\n", predict);

	return 0;
}