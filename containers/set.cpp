/*
    Container to store unique elements whose order follows a strict weak ordering indicated by comparison object.
    Set is slower than unordered_set, but allowing direct iteration through elements based on their order.
    A set is implemented as binary seach tree.
*/

#include <bits/stdc++.h>
using namespace std;

bool fn_pt(int lhs, int rhs) {
    return lhs<rhs;
}

struct classcomp {
    bool operator() (const int& lhs, const int& rhs) const {
        return lhs<rhs;
    }
};

int main() {
    // init
    set<int> first;
    int myints[]= {10, 20, 30, 40, 50};
    set<int> second (myints,myints + 5); 
    set<int> third (second);
    set<int> fourth (second.begin(), second.end());

    // custom order
    std::set<int, classcomp> fifth; // class as Compare
    set<int, bool(*)(int, int)> sixth (fn_pt); // function pointer as compare

    // insert
    fifth.insert(44); // O(logn)

    // find
    auto iter = fifth.find(44); // O(logn): return the iterator of the value, else return set::end

    // erase
    // second.erase(second.begin()); // O(1): erase(position)
    second.erase(20); // O(logn): erase(val)
    // second.erase(second.begin(), second.end()); // O(n): erase(first, last) linear in distance of first and last

    // lower bound and upper bound
    auto ilow = second.lower_bound(10); // O(logn): first value >= 10
    auto iup = second.upper_bound(30); // O(logn): first value > 30

    // equal range
    pair<set<int>::iterator, set<int>::iterator> ret = second.equal_range(30); // Only one element since elements are unique

    // iterate
    set<pair<int, int>> seventh;
    seventh.insert({1, 2});
    for (auto iter = seventh.begin(); iter != seventh.end(); ++iter) {
        cout << (*iter).first << " " << (*iter).second;
    }
    cout << '\n';
    for (auto iter = fifth.begin(); iter != fifth.end(); ++iter) {
        cout << *iter;
    }
    return 0;
}