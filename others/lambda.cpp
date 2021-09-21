/*
    An example of a lambda expression: [&](int a){ return a < x; }. In this expression, [&] is a 
    capture list specifying that local names used (such as x) will be accessed through references. 
    Had we wanted to "capture" on x, we could have said so: [&x]. Had we wanted to give the 
    generated object a copy of x, we could have said so: [x]. Capture nothing is [], 
    capture all local names used by references is [&], and capture all local names used by values 
    is [=].
*/

#include <bits/stdc++.h>
using namespace std;

template<typename C, typename Op>
void for_all(C& c, Op op) {
    for (auto &x: c) op(*x);
}

class Shape {
    void draw() {}
    void rotate(int deg) {}
};

Shape* read_shape(istream cin) {
    Shape *random;
    return random;
}

// void user() {
//     vector<unique_ptr<Shape>> v;
//     while (cin) v.push_back(read_shape(cin));
//     // Pass an lambda expression as a function objects to template
//     for_all(v, [](Shape& s){ s.draw(); });
//     for_all(v, [](Shape& s){ s.rotate(45); });
// }

int main() {
    return 0;
}