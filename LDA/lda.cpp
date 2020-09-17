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
		for (auto w = (*d).begin(); w != (*d).end(); w++) {
			word_set.insert(*w);
		}
	}
	V = word_set.size();
	M = doc.size();
	cout << "topic number" << K << endl;
	cout << "doc number" << M << endl;
	cout << "word number" << V << endl;
	boost::uniform_int<int> random(0, K);
	for (int i = 0; i < K; i++) {
		vector<int> temp(V, 0);
		vector<double> temp2(V, 0.0);
		nkt.push_back(temp);
		nktSum.push_back(0);
		phi.push_back(temp2);
	}
	for (auto d = doc.begin(); d != doc.end(); d++) {
		vector<int> temp(d->size(), 0);
		z.push_back(temp);
		vector<int> temp2(K, 0);
		nmk.push_back(temp2);
		nmkSum.push_back(0);
		vector<double> temp3(K, 0.0);
		theta.push_back(temp3);
		for (int i = 0; i < d->size(); i++) {
			int initTopic = random(rng);
			z.back().at(i) = initTopic;
			nmk.back().at(initTopic)++;
			nkt.at(initTopic).at(d->at(i))++;
			nktSum.at(initTopic)++;
		}
		nmkSum.back() = d->size();
	}
}
void LDA::inference() {

}
void LDA::update() {

}