/*
    The elements are sorted according to the comparison object. 
*/

#include <bits/stdc++.h>
using namespace std;

bool fncomp (char lhs, char rhs) {
    return lhs<rhs;
}

struct classcomp {
    bool operator() (const char& lhs, const char& rhs) const {
        return lhs<rhs;
    }
};

int main() {
    // init
    map<char,int> first;
    map<char,int> second (first.begin(),first.end());

    map<char,int> third (second);
    map<char,int,classcomp> fourth; // class as Compare

    bool(*fn_pt)(char,char) = fncomp;
    map<char,int,bool(*)(char,char)> fifth (fn_pt);

    // find
    auto iter = first.find('b'); // O(logn)

    // lower bound, upper bound, equal_range
    first['b'] = 1;
    auto ilow = first.lower_bound('b');
    // cout << ilow->first << " " << ilow->second;
    auto iup = first.upper_bound('b');

    // equal_range
    pair<std::map<char,int>::iterator,std::map<char,int>::iterator> ret = first.equal_range('b');
    cout << ret.first->first << " " << ret.first->second << '\n';

    // erase
    first.erase(first.begin()); // O(1): erase(position), amortized constant
    first.erase('a'); // O(logn): erase(val)
    first.erase(first.begin(), first.end()); // O(n): erase(first, last)

    return 0;
}