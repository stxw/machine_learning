/* K-近邻算法 */
#include <stdio.h>
#include <math.h>
#include <map>

using namespace std;

typedef struct point
{
    double x, y;
} point;

int main()
{
    int i, m = 4;
    point inp;
    point data[100] = {{1.0, 1.1}, {1.0, 1.0}, {0, 0}, {0, 0.1}};
    char labs[100] = {'A', 'A', 'B', 'B'};
    map<char, double> mp;

    scanf("%lf%lf", &inp.x, &inp.y);
    for(i = 0; i < m; i++)
    {
        double dist = pow(inp.x - data[i].x, 2) + pow(inp.y - data[i].y, 2);
        mp[labs[i]] += dist;
    }

    char ans;
    double mind = 1e9;
    for(map<char, double>::iterator iter = mp.begin(); iter != mp.end(); iter++)
    {
        if(mind > iter->second)
        {
            mind = iter->second;
            ans = iter->first;
        }
    }
    printf("%c\n", ans);
    return 0;
}