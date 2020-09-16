#include<iostream>
#include<string>
#include"csv.h"
#include<boost/program_options.hpp>
using namespace std;
namespace po = boost::program_options;
int main(int argc, char const* argv[]) {
	double alpha;
	double beta;
	int iterNumber;
	int infNumber;
	int topicNumber;
	po::options_description opt("all options");
	opt.add_options()
		("alpha", po::value<double>(&alpha)->default_value(0.1))
		("beta", po::value<double>(&beta)->default_value(0.1))
		("iter", po::value<int>(&iterNumber)->default_value(100))
		("inf", po::value<int>(&infNumber)->default_value(10))
		("topic", po::value<int>(&topicNumber)->default_value(100));
	po::variables_map vm;
	try {
		po::store(parse_command_line(argc, argv, opt), vm);
	}
	catch (...) {
		std::cout << "input param wrong!!!" << endl;
	}
	po::notify(vm);
	cout << topicNumber << endl;
	cout << "hello world" << endl;
}