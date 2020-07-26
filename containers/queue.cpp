/*
    FIFO container. 
*/

#include <bits/stdc++.h>
using namespace std;

int main() {
    queue<int> q;

    // insert
    for (int i = 0; i < 10; ++i) {
        q.push(i); // O(1) amortized. One call to push_back on the underlying container
    }

    // top element of queue: oldest
    cout << q.front() << '\n'; // O(1): calling front on the underlying container

    // last element of queue: newest
    cout << q.back() << '\n'; // O(1): calling back on the underlying container

    // remove top element
    q.pop(); // O(1): calling pop_front on the underlying container
    return 0;
}