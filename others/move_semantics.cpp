/*
    move() is simply a cast to an rvalue (*). It produces an rvalue for its argument, so that the object referred can be moved from. move() is used to tell the compiler that an object will not be used anymore in a context, so that its value can be moved
    and empty object left behind. Eg: swap()

    forward(): produce rvalue from an rvalue for lvalue (**). Use for "perfect forwarding" of an argument from 1 function to the other. Example: make_shared<T>(X)
*/

#include <bits/stdc++.h>
using namespace std;

// (*)
template<typename T>
remove_reference<T>&& move(T&& t) noexcept {
    return static_cast<remove_reference<T>&&>(t);
}

// (**)
template <class T>
inline T&& forward_r(typename std::remove_reference<T>::type& t) noexcept {
    return static_cast<T&&>(t);
}

template <class T>
inline T&& forward_l(typename std::remove_reference<T>::type&& t) noexcept {
    static_assert(!std::is_lvalue_reference<T>::value,
                  "Can not forward an rvalue as an lvalue.");
    return static_cast<T&&>(t);
}

int main() {
    int i = 7;
    forward_r<int>(i); // lvalue forward
    forward_l<int>(7); // rvalue forward
}

// class Vector {
//     Vector(const Vector&a); // Copy constructor
//     Vector& operator=(const Vector&a); // Copy assignment

//     Vector(Vector&&a); // Move Constructor
//     Vector& operator=(Vector&&a); // Move assignment
// };