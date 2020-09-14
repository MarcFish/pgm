#pragma once
class LDA {
private:
	double alpha;
	double beta;
	int iterNumber;
	int topicNumber;
public:
	LDA(double alpha=0.5,double beta=0.1,int iterNumber=100,int topicNumber=100);
	void initModel();
};