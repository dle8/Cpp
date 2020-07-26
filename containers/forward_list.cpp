/*
    Singly-linked list: sequence containers for O(1) insert and erase operations anywhere within the sequence.
    Singly-linked list only contains a link to the next element, while list contains two links for both next and previous
    elements.

    drawbacK: 
    - forward_list can not access to the element using index, it has to iterate to that index to access the element.
    - it lacks O(1) method to know the size. Only know size using distance algorithm with its begin and end, taking O(n)
    
*/

#include <bits/stdc++.h>
using namespace std;

int main() {
    forward_list<int> mylist = {77, 2, 16};
    forward_list<pair<int, int>> plist = {{1, 1}};

    // insert
    mylist.push_front (19); // O(1): insert at the beginning by either copying or moving an existing obj to container
    plist.emplace_front(10, 10); // O(1): insert at the beginning by constructing in place using args as the arguments for its construction


    // delete
    mylist.pop_front(); // O(1): remove the first element

    // iterate 
    for (int& x: mylist) std::cout << ' ' << x;
    cout << '\n';
    for (auto& x: plist) cout << x.first << " " << x.second << '\n';

    return 0;
}