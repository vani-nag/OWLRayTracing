#include<iostream>
using namespace std;

void addTriangles(float x, float y, float radius, int index)
{	
	float y_pos_max, y_neg_max, x_pos_max, x_neg_max;
	y_pos_max = y + radius;
	y_neg_max = y - radius;	

        x_pos_max = x + radius;
        x_neg_max = x - radius;

	//vertices
	std::cout<<x_neg_max<<", "<<y+radius+radius-1<<"\n";
	std::cout<<x_neg_max<<", "<<y-radius-radius+1<<"\n";
	std::cout<<x_pos_max+radius+1<<", "<<y<<"\n";
}

int main()
{
	addTriangles(0,1,4,1);
	return 0;
}	

