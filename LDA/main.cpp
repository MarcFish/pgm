#include<iostream>
#include<string>
#include<regex>
#include<vector>
#include <iterator>
#include <fstream>
#include <sstream>
#include<boost/program_options.hpp>
#include"lda.h"
using namespace std;
namespace po = boost::program_options;
vector<vector<int>> doc;

std::vector<int> getLine(std::istream& str)
{
	std::vector<int>   result;
	std::string                line;
	std::getline(str, line);

	std::stringstream          lineStream(line);
	std::string                cell;

	while (std::getline(lineStream, cell, ','))
	{
		result.push_back(stoi(cell));
	}
	return result;
}

int main(int argc, char const* argv[]) {
	// cmdline parse
	double alpha;
	double beta;
	int iterNumber;
	int infNumber;
	int topicNumber;
	string docfile;
	po::options_description opt("all options");
	opt.add_options()
		("alpha", po::value<double>(&alpha)->default_value(0.1))
		("beta", po::value<double>(&beta)->default_value(0.1))
		("iter", po::value<int>(&iterNumber)->default_value(100))
		("inf", po::value<int>(&infNumber)->default_value(10))
		("topic", po::value<int>(&topicNumber)->default_value(100))
		("file", po::value<string>(&docfile)->default_value("../data/reuter.csv"));
	po::variables_map vm;
	try {
		po::store(parse_command_line(argc, argv, opt), vm);
	}
	catch (...) {
		std::cout << "input param wrong!!!" << endl;
	}
	po::notify(vm);
	cout << "read doc file" << endl;
	ifstream docStream;
	docStream.open(docfile);
	while (!docStream.eof()) {
		doc.push_back(getLine(docStream));
	}
	docStream.close();
	doc.pop_back();
	LDA lda = LDA(doc);
	cout << "init model" << endl;
	lda.initModel();
	lda.inference();
}