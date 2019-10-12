#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <math.h>

using namespace std;

int cluster[2000];
int nearest(double center_x[],double center_y[],double x,double y,int k){
	double dis[1000];
	int s;
	for(int i=0;i<k;i++)
		dis[i]=sqrt((x-center_x[i])*(x-center_x[i])+(y-center_y[i])*(y-center_y[i]));
	s=0;
	for(int i=1;i<k;i++){
		if(dis[i]<dis[s])s=i;
	}
	return s;
}

void k_mean(double center_x[],double center_y[],double x[],double y[],int count,int k){
	
	int next = k;
	double center_sum_x[1000];
	double center_sum_y[1000];
	double center_bf_x[1000];
	double center_bf_y[1000];
	int center_count[1000];
	while(next>0){
		next=k;
		for(int i=0;i<k;i++){
			center_bf_x[i]=center_x[i];
			center_bf_y[i]=center_y[i];
			center_sum_x[i]=0;
			center_sum_y[i]=0;
			center_count[i]=0;
		}
		for(int i=1;i<count;i++){
			cluster[i] = nearest(center_x,center_y,x[i],y[i],k);
			center_sum_x[cluster[i]]+=x[i];
			center_sum_y[cluster[i]]+=y[i];
			center_count[cluster[i]]++;
		}
		for(int i=0;i<k;i++){
			center_x[i]=center_sum_x[i]/center_count[i];
			center_y[i]=center_sum_y[i]/center_count[i];
			if(sqrt((center_x[i]-center_bf_x[i])*(center_x[i]-center_bf_x[i])+(center_y[i]-center_bf_y[i])*(center_y[i]-center_bf_y[i]))<1e-3)next--;
		}
	}
	
}
 
int main()
{
	ifstream fin("data_noah.csv"); 
	string line; 
	double center_x[1000],center_y[1000];
	double x[2000],y[2000];
	int count=0;
	while (getline(fin, line))   
	{
		istringstream sin(line); 
		vector<string> fields; 
		string field;
		while (getline(sin, field, ','))
		{
			fields.push_back(field);
		}
		x[count] = atof((fields[13]).c_str()); 
		y[count] = atof((fields[14]).c_str());  
		count++;
	}
	for(int k=1;k<10;k++){
		for(int i=0;i<k;i++){
			center_x[i]=x[i+1];
			center_y[i]=y[i+1];
		}
		double max_x=x[1],max_y=y[1],min_x=x[1],min_y=y[1];
		double cost=0;
		
		k_mean(center_x,center_y,x,y,count,k);
		for(int i=1;i<count;i++){
			cost+=((x[i]-center_x[cluster[i]])*(x[i]-center_x[cluster[i]])+(y[i]-center_y[cluster[i]])*(y[i]-center_y[cluster[i]]));
			
		}
		printf("k=%d cost = %lf\n",k,cost);	
	
	}
	
	return 0;
}
