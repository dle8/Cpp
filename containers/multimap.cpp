/*
    - Containers that store elements in specific order, and there can be multiple elements with same value.
    - Implemented as binary search trees.
*/

#include <bits/stdc++.h>
using namespace std;

int main() {
    multiset<int> mymultiset;

    // insert
    for (int i = 1; i <= 5; ++i) {
        for (int j = 1; j <= 2; ++j) {
            mymultiset.insert(i); // O(logn)
        }
    }

    // find
    auto iter = mymultiset.find(1); // O(logn): An iterator to the element, if val is found, or multiset::end otherwise.

    // count
    size_t cnt = mymultiset.count(3); // O(logn)

    // custom sort
    int tmp[] = {3, 4, 5};
    auto cmp = [&tmp](const int&x, const int&y) {
        return tmp[x] < tmp[y];
    };
    multiset<int, decltype(&cmp)> ms(&cmp);

    // lower bound, upper bound, equal_range
    auto ilow = mymultiset.lower_bound(30); // O(logn)
    auto iup = mymultiset.lower_bound(40); // O(logn)
    pair<multiset<int>::iterator, multiset<int>::iterator> ret = mymultiset.equal_range(30); // O(logn)

    // erase: remove all elements with that value
    mymultiset.erase(mymultiset.begin()); // O(1): erase(position)
    mymultiset.erase(2); // O(n): linear in number of elements removed. O(logn) for one
    mymultiset.erase(mymultiset.begin(), mymultiset.end()); // O(n): erase(first, last) linear 
    return 0;
}