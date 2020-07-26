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
    cout << s.top();

    // remove top element
    s.pop();
    return 0;
}