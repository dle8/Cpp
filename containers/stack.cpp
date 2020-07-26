/*
    LIFO container. 
*/

#include <bits/stdc++.h>
using namespace std;

int main() {
    stack<int> s;

    // insert
    for (int i = 0; i < 10; ++i) {
        s.push(i); // O(1) amortized. One call to push_back on the underlying container
    }

    // top element
    cout << s.top(); // O(1): calling back on the underlying container

    // remove top element
    s.pop(); // O(1): calling pop_back on the underlying container
    return 0;
}