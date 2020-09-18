#include "lda.h"
#include<set>
#include<vector>
#include<algorithm>
#include<random>
#include<boost/random/uniform_int.hpp>
#include<boost/random.hpp>
using namespace std;
boost::random::mt19937 rng;
void LDA::initModel() {
	set<int> word_set;
	for (auto d = doc.begin(); d != doc.end(); d++) {
		vector<int> temp(d->size(), 0);
		z.push_back(temp);
		vector<int> temp2(K, 0);
		nmk.push_back(temp2);
		nmkSum.push_back(0);
		vector<double> temp3(K, 0.0);
		theta.push_back(temp3);
		for (auto w = (*d).begin(); w != (*d).end(); w++) {
			word_set.insert(*w);
		}
	}
	V = word_set.size();
	M = doc.size();
	cout << "topic number" << K << endl;
	cout << "doc number" << M << endl;
	cout << "word number" << V << endl;
	boost::uniform_int<int> random(0, K-1);
	for (int i = 0; i < K; i++) {
		vector<int> temp(V, 0);
		vector<double> temp2(V, 0.0);
		nkt.push_back(temp);
		nktSum.push_back(0);
		phi.push_back(temp2);
	}
	cout << "init" << endl;
	for (int j = 0; j < M;j++) {
		for (int i = 0; i < doc.at(j).size(); i++) {
			int initTopic = random(rng);
			z.at(j).at(i) = initTopic;
			nmk.at(j).at(initTopic)++;
			nkt.at(initTopic).at(doc.at(j).at(i))++;
			nktSum.at(initTopic)++;
		}
		nmkSum.at(j) = doc.at(j).size();
	}
}
void LDA::inference() {
	for(int i=0;i<iterNumber;i++){
		cout << "iter:" << i << endl;
		if(i>=infNumber){
			update();
		}
		for(int m=0;m<M;m++){
			int N = doc.at(m).size();
			for(int n=0;n<N;n++){
				int newTopic = sampleTopic(m,n);
				z.at(m).at(n) = newTopic;
			}
		}
	}

}
int LDA::sampleTopic(int m, int n){
	int oldTopic = z.at(m).at(n);
	nmk.at(m).at(oldTopic)--;
	nkt.at(oldTopic).at(doc.at(m).at(n))--;
	nmkSum.at(m)--;
	nktSum.at(oldTopic)--;
	vector<double>p(K);
	for (int i = 0; i < K; i++) {
		p.at(i) = (nkt.at(i).at(doc.at(m).at(n)) + beta) / (nktSum.at(i) + V * beta) * (nmk.at(m).at(i) + alpha) / (nmkSum.at(m) + K * alpha);
	}
	std::discrete_distribution<int> distribution(begin(p), end(p));
	std::default_random_engine generator;
	int newTopic = distribution(generator);
	nmk.at(m).at(newTopic)++;
	nkt.at(newTopic).at(doc.at(m).at(n))++;
	nmkSum.at(m)++;
	nktSum.at(newTopic)++;
	return newTopic;
}
void LDA::update() {
	for (int k = 0; k < K; k++) {
		for (int v = 0; v < V; v++) {
			phi.at(k).at(v) = (nkt.at(k).at(v) + beta) / (nktSum.at(k) + V * beta);
		}
	}
	for (int m = 0; m < M; m++) {
		for (int k = 0; k < K; k++) {
			theta.at(m).at(k) = (nmk.at(m).at(k) + alpha) / (nmkSum.at(m) + K * alpha);
		}
	}

}