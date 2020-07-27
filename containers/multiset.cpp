/*
    - Containers that store elements following a specific order, and where multiple elements can have equivalent values.
    - Implemented as binary search trees.
*/
#include <bits/stdc++.h>
using namespace std;

bool fncomp (int lhs, int rhs) {
    return lhs<rhs;
}

struct classcomp {
    bool operator() (const int& lhs, const int& rhs) const {
        return lhs<rhs;
    }
};

int main() {
    multiset<int> first;
    int myints[]= {10, 20, 30, 20, 20};
    multiset<int> second (myints,myints+5);       // pointers used as iterators
    multiset<int, classcomp> fifth; // class as Compare
    bool(*fn_pt)(int, int) = fncomp;
    multiset<int, bool(*)(int, int)> sixth (fn_pt); // function pointer as Compare

    // count
    cout << second.count(73) << '\n'; // O(logn): Logarithmic in size and linear in the number of matches.
    
    // find
    auto iter = second.find(10); // O(logn)

    // lower bound, upper bound, equal range
    auto ilow = second.lower_bound(10); // O(logn)
    auto iup = second.upper_bound(10); // O(logn)
    pair<multiset<int>::iterator, multiset<int>::iterator> ret = second.equal_range(20); // O(logn)

    // erase
    second.erase(second.begin())/ // O(1)
    second.erase(10); // O(logn)
    second.erase(second.begin(), second.end()); // O(n): linear in distance between first and last
    return 0;
}
