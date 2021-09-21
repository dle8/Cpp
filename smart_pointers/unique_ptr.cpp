/*
    - The use of smart pointers is to prevent memory leaks caused by careless programming. 
    unique_ptr ensures that is object is properly destroyed whichever way the program exit, 
    either by exitting naturally or an error is thrown, by automatically calling delete or 
    delete[] on the object. It further uses include passing free-store allocated objects in and 
    out of functions (**)

    Unique_ptr controls the lifetime of an object using RAII and relies on move semantics to 
    make return simple and efficient. (**).

    RAII: Resource Acquisition Is Initialization eliminates using new and delete by acquiring 
    resources in a constructor and release it using destructor.
*/

#include <bits/stdc++.h>

using namespace std;

class X {

public:
    int x;

    X(int i): x(i) {
        cout << x << '\n';
    }
};

void f(int i, int j) { // X* vs unique_ptr<X>
    X* p = new X{i};
    unique_ptr<X> sp {new X{j}};
    delete(p);
}

unique_ptr<X> make_X(int i) {
    return unique_ptr<X>{new X{i}};
}

int main() {
    f(1, 2);
    unique_ptr<X> ptr = make_X(3);
}