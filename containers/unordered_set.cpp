/*
    Containers that store unique elements in no particular order, and which allow for fast retrieval of individual 
    elements based on their value. Elements are organized into buckets depending on their hash values to allow for 
    fast access to individual elements directly by their values


*/

#include <bits/stdc++.h>
using namespace std;

int main() {
    unordered_set<string> second ( {"red","green","blue"} );
    unordered_set<string> fourth ( second ); 
    unordered_set<string> sixth ( fourth.begin(), fourth.end() );

    // insert
    string str = "abcd";
    second.insert(str); // O(1) average: copy insertion. O(n): worst
    second.insert(str + "dist"); // O(1) average: move insertion. O(n) worst
    second.insert(fourth.begin(), fourth.end()); // O(n): range insertion
    second.insert ( {"purple","orange"} );  // O(n): initializer list insertion
    
    // find
    auto iter = second.find("red"); // O(1) average. O(n) worst

    // erase
    second.erase("red"); // O(1) average.
    second.erase(second.begin()); // O(1)
    second.erase(second.find("green"), second.end()); // O(n): linear in container size

    // iterate
    for (const string& x: sixth) cout << " " << x;
    return 0;
}