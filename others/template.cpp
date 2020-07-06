#include <bits/stdc++.h>

template <class T>
class Widget {
public:
    void setName() {}

    // Template class with template method
    template<class M>
    void setname(M data) {}
};

// Use template class with a specialized method
template<>
void Widget<int>::setName() { // template class specialization of a member

} 

// Template class with a class partial specialization. Only templated classes may be
// partially specialized. Templated functions must be fully specialized
template <class X>
class Widget<std::vector<X>> {

};

Widget<int> foo1; // T is int
Widget<std::vector<int>> myVar; // X is int
