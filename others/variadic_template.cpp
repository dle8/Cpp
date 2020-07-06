/*
    A variadic template is a template that is defined to accept an arbitrary number of arguments of arbitraty types. The key to implementing a 
    variadic template is to note that when you pass a list of arguments to it, you can separate the first argument from the rest. Here we do
    something for the first argument(the head) and then recursively call f() with the rest of the arguments (the tail). The ellipsis, ..., is
    used to indicate "the rest" of a list. Eventually, of course, "tail" will become empty and we need a separate function to deal with that.
*/

#include <bits/stdc++.h>
using namespace std;

void f() {} // do nothing

// In a real program, g(head) does whatever we wanted to do to each argument. For example, use g to print out argument:
template<typename T>
void g(T x) {
    cout << x << " ";
}

template<typename T, typename... Tail>
void f(T head, Tail... tail) {
    g(head); 
    f(tail...);
}

int main() {
    cout << "first: ";
    f(1, 2.2, "hello"); // this would call f(1, 2.2, "hello") which will call f(2.2, "hello") which will call f("hello") which will call f()

    cout << "\n second: ";
    f(0.2, 'c', "yuck!", 0, 1, 2);
    cout << '\n';
    return 0;
}
 
