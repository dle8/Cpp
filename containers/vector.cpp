/*
    Vectors use contiguous storage locations for their elements. Vectors use a dynamically allocated array to store their 
    elements. This array may need to be reallocated in order to grow in size -> allocating a new array and moving all 
    elements to it. Since this is expensive, vectors allocate more size than needed so th reallocations should only 
    happen at logarithmically growing intervals of size so that complexity of inserting is amortized constant.

    vector can act as a stack
*/
#include <bits/stdc++.h>
using namespace std;

int main() {
    // init 
    vector<int> first;                                // empty vector of ints
    vector<int> second (4,100);                       // four ints with value 100
    vector<int> third (second.begin(),second.end());  // iterating through second
    vector<int> fourth (third);                       // a copy of third

    // the iterator constructor can also be used to construct from arrays:
    int myints[] = {16,2,77,29, 100, 24};
    vector<int> fifth (myints, myints + sizeof(myints) / sizeof(int) );
    
    // access front & back element -  O(1)
    int front = fifth.front();
    int back = fifth.back();

    // insert: O(n): linear on number of elements inserted + number of elements after position (moving)
    fifth.insert(fifth.begin(), 10); // insert at the beginning
    fifth.insert(fifth.begin() + 2, 20); // insert at index [2], so previous index [2] now becomes index [3]
    fifth.insert(fifth.begin() + 1, fourth.begin(), fourth.end()); // insert at index [1] the whole forth vector

    int arr = {501, 502, 503};
    fifth.insert(fifth.begin(), arr, arr + 2); // insert at the beginning the first 2 elements of arr
    
    // erase elements: O(n) - linear on number of element erase + number of elements after last element deleted (moving)
    fifth.erase(fifth.begin()); // erase the first element
    fifth.erase(fifth.begin() + 2); // erase the third element
    fifth.erase(fifth.begin() + 1, fifth.begin() + 3); // erase elements with index from [1, 3), or [1, 2] of 0-based index

    // remove last element
    fifth.pop_back(); // O(1)
    return 0;
}