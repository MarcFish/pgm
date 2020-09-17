#include "lda.h"
#include<set>
#include<vector>
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
void LDA::sampleTopic(int m, int n){
	int oldTopic = z.at(m).at(n);
	nmk.at(m).at(oldTopic)--;
	nkt.at(oldTopic).at(doc.at(m).at(n))--;
	nmkSum.at(m)--;
	nktSum.at(oldTopic)--;
}
void LDA::update() {

}