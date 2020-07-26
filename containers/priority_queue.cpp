/*
    Heap: elements are compared based on strict weak ordering criterion
*/

#include <bits/stdc++.h>
using namespace std;

// template <class T, class Container = vector<T>, class Compare = less<typename Container::value_type> > 
// class priority_queue;

// Override compare function
class Person {
public:
    string name;
    int age;

    Person(string name, int age): name(name), age(age) {}
};

bool operator < (const Person&p1, const Person&p2) {
    return p1.age > p2.age; // Sort by increasing age
}


struct cmp {
    bool operator()(const Person&p1, const Person&p2) {
        return p1.age > p2.age; // Sort by increasing age
    }
};

int main() {
    // max heap
    priority_queue<int> max_heap;

    // min heap
    priority_queue<int, vector<int>, greater<int> > min_heap;

    Person p1("abc", 15);
    Person p2("abc", 10);
    
    // custom compare without using struct
    priority_queue<Person, vector<Person>, less<vector<Person>::value_type> > pq_no_struct;

    // custom compare with struct
    priority_queue<Person, vector<Person>, cmp> pq_struct;

    // custom compare with lambda - works with C++11 and above
    auto compare = [](const Person&p1, const Person&p2) {
        return p1.age > p2.age;
    };
    priority_queue<Person, vector<Person>, decltype(compare)> pq_lambda(compare);

    // push
    pq_lambda.push({"abc", 10}); // O(logn)
    pq_lambda.push({"bcd", 15}); // O(logn)

    while (!pq_lambda.empty()) {
        // get the top element
        cout << pq_lambda.top().name << " " << pq_lambda.top().age << '\n'; // O(1)

        // pop
        pq_lambda.pop(); // O(logn)
    }

    return 0;
}