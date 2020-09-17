#include<vector>
using namespace std;

class LDA {
private:
	double alpha;
	double beta;
	int iterNumber;
	int infNumber;
	vector<vector<int>> nmk;
	vector<vector<int>> nkt;
	vector<int> nmkSum;
	vector<int> nktSum;
	vector<vector<double>> phi;
	vector<vector<double>> theta;
	vector<vector<int>> z;
	vector<vector<int>> doc;
	int V, K, M;  // vocabulary, topic, document
public:
	LDA(vector<vector<int>> doc, double alpha = 0.5, double beta = 0.1, int iterNumber = 100, int infNumber = 10, int topicNumber = 100):
		alpha(alpha), beta(beta), iterNumber(iterNumber), infNumber(infNumber), K(topicNumber), doc(doc) {}
	void initModel();
	void update();
	void inference();
};