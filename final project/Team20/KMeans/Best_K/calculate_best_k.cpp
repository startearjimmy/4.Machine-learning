#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <math.h>
#define data_max 20000
using namespace std;

int cluster[data_max];
int nearest(double center_x[],double center_y[],double x,double y,int k){
	double dis[data_max];
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
	double center_sum_x[data_max];
	double center_sum_y[data_max];
	double center_bf_x[data_max];
	double center_bf_y[data_max];
	int center_count[data_max];
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
void splitStr2Vec(string s, vector<string>& buf){
	while(buf.size()>0)
		buf.pop_back();
	int current = 0; //最初由 0 的位置開始找
	int next;
	while (1)
	{
		next = s.find_first_of("/", current);
		if (next != current)
		{
			string tmp = s.substr(current, next - current);
			if (tmp.size() != 0) //忽略空字串
				buf.push_back(tmp);
		}
		if (next == string::npos) break;
		current = next + 1; //下次由 next + 1 的位置開始找起。
	}
}

int main(){
	ifstream fin("./Resources/TSMC.csv"); 
	string line; 
	double center_x[data_max],center_y[data_max];
	double x[data_max],y[data_max];
	int count=0;
	string test;
    vector<string> aryLine;
    string lastLine;
    double sum = 0;
    double avg = 0;
    int sum_count = 0;
    int start_site = 0;
    bool skip = true;
	while (getline(fin, line)){
		if(skip){
			skip = false;
			continue;
		}
		istringstream sin(line); 
		vector<string> fields; 
		string field;
		while (getline(sin, field, ','))
		{
			fields.push_back(field);
		}
		x[count] = atof((fields[4]).c_str()); 
		y[count] = atof((fields[1]).c_str())-x[count];  
		
		splitStr2Vec((fields[0]).c_str(),aryLine); 
		
        sum += x[count];
        sum_count++;
        if (count != 0){
            if ((aryLine[1] == "01" && lastLine == "12")) {
                avg = sum / sum_count;
                for (int j = start_site; j < count; j++)
                    x[j] -= avg;
                sum = x[count];
                start_site = count;
                sum_count = 1;

            }
        }
        
		count++;
		lastLine = aryLine[1];
		if(count==data_max)break;
	}
	
	avg = sum / sum_count;
    for (int j = start_site; j < count; j++)
        x[j] -= avg;
        
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
