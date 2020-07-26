/*
    Doubly-linked lists: constant time insert and erase anywhere + iteration in both directions.
    A list stores each of the elements they contain in different and unrelated storage locations. It lacks direct access 
    to the elements by their position.
*/

#include <bits/stdc++.h>
using namespace std;

int main() {
    // init
    list<int> first;                                // empty list of ints
    list<int> second (4,100);                       // four ints with value 100
    list<int> third (second.begin(),second.end());  // iterating through second
    list<int> fourth (third);                       // a copy of third

    int myints[] = {16,2,77,29};
    list<int> fifth (myints, myints + sizeof(myints) / sizeof(int) );
    list<int>::iterator iter1 = myints.begin();
    list<int>::iterator iter2 = myints.begin();

    // push
    for (int i = 0; i < 10; ++i) { 
        myints.push_back(i * 2);  // O(1)
        myints.push_front(i * 3); // O(1)
    } 

    // insert: O(n) in number of elements inserted
    myints.insert(iter1, 2, 20); // insert two numbers at position of iter1. now new iter1 = iter1 + 2


    // delete
    myints.pop_front(); // O(1)
    myints.pop_back(); // O(1)

    // erase: O(n) - linear in the number of element erased
    ++iter1;
    advance(iter2, 6);
    iter = myints.erase(iter); // return the pointer to the current position but value is now replaced by value of the next element
    myints.erase(iter1, iter2);

    // iterate
    for (list<int>::iterator it = fifth.begin(); it != fifth.end(); it++) {
        cout << *it << ' ';
    }

    return 0;
}