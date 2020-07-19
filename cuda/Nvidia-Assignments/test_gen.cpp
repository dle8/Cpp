#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

const int NUM_TEST = 100;

bool diff(string file1, string file2) {
  string ans1, ans2;
  ifstream out(file1.c_str());
  getline(out, ans1);
  ifstream ans(file2.c_str());
  getline(ans, ans2);
  cout << ans1 << '\n';
  cout << ans2 << '\n';
  return (ans1 == ans2);
}

int main() {
  bool stopped = false;
  for (int current_test = 1; current_test <= NUM_TEST; ++current_test) {
    srand(time(NULL));
    ofstream inp("input.txt");

    // Input making goes here.
    int homes = rand() % 10001; 
    inp << homes << '\n';
    int max_candy = rand() % 1001;
    inp << max_candy << '\n';
    for (int i = 0; i < homes; ++i) {
        int pieces = rand() % 10;
        inp << pieces << '\n';
    }

    inp.close();
    system("g++ -o run trick_or_treat.cpp");
    system("./run >> output.txt");
    system("g++ -o run -fopenmp trick_or_treat_openmp.cpp");
    system("./run >> output_openmp.txt");

    if (diff("output.txt", "output_openmp.txt") == 0) {
        cout << "Test #" << current_test << ": DIFFERENCE BETWEEN SEQUENTIAL AND OPENMP ANSWERS!";
        stopped = true;
    }
    else cout << "Test #" << current_test << ": SAME ANSWERS";
    cout << '\n';
    system("rm -rf output.txt output_openmp.txt run");
    if (stopped) break;
  }
  return 0;
}