/* 线性回归 */
#include <stdio.h>
#include <math.h>

struct data
{
	double x, y;
};

int main()
{
	double th0, th1, alip;
	int i, j, k, m;
	data dap[100] = {{0.5, 1.9}, {1.7, 2.8}, {1.9, 2.2}, {2.4, 3.6},
					{3.0, 3.2}, {3.2, 2.3}, {3.5, 3.0}, {3.6, 3.8}};
	
	m = 8;
	th0 = 0;
	th1 = 0;
	alip = 0.1;

	for(k = 0; k < 1000; k++)
	{
		double sum0 = 0.0, sum1 = 0.0;
		for(i = 0; i < m; i++)
		{
			sum0 += (th0 + th1 * dap[i].x - dap[i].y) * 1.0;
			sum1 += (th0 + th1 * dap[i].x - dap[i].y) * dap[i].x;
		}
		th0 = th0 - sum0 * alip / m;
		th1 = th1 - sum1 * alip / m;

		double sum_p2 = 0.0;
		for(i = 0; i < m; i++)
		{
			sum_p2 += pow((th0 + th1 * dap[i].x - dap[i].y), 2.0);
		}
		printf("J%d=%lf\n",k, sum_p2 / 2 / m);
	}
	printf("y = %lfx + %lf\n", th1, th0);
	return 0;
}