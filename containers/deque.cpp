/*
    Double-ended queue: sequence containers with dynamic sizes that can be expanded or contracted on both ends. Unlike 
    vectors, deque's elements are not in contiguous storage locations: accessing elements in a deque by offsetting a pointer 
    to another element -> undefined behavior. Elements of a deque can be scattered in different chunks of storage, with 
    deque keeping the necessary information to provide direct access to its elements in constant time by iterators.
*/


#include <bits/stdc++.h>
using namespace std;

int main() {
    deque<int> d = {7, 5, 16, 8};
    
    // insert - all O(1)
    d.push_front(13); // O(1)
    d.push_back(25); // O(1)

    // delete 
    d.pop_back(); // delete the last elements O(1)
    d.pop_front(); // delete the front elements O(1)

    // iterator through elements
    for (deque<int>::iterator iter = d.begin(); iter != d.end(); ++iter) {
        cout << *iter << " ";
    }
    cout << '\n';

    // erase: O(n): linear on the number of element erased + number of elements after that depends on lib implementation
    d.erase(d.begin(), d.begin() + 2); // erase the first 2 elements

    // access elements
    cout << d.front() << '\n';
    cout << d.back() << '\n';
    return 0;
}