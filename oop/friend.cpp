/*
    OOP in C++ can be used using "class" and "struct" in C++. "Struct" has default public members and bases, while "Class" has default private
    member and bases. Both can have a mixtureof public, protected, and private members, can use inheritance and member functions.

    A friend class can access private and protected members of other class in which it's declared as friend. It's sometimes useful to allow a
    particular class to access private members of other class. For example, a LinkedList class maybe allowed to access private members of Node.
    Friend function, like friend class, gives special grant to access private and protected members. A friend function can be: a method of
    another class, or a global function.

    Friend should be used only for limited purposes.
    Friendship is not mutual. If class A is a friend of B, B doesn't become a friend of A automatically
    Friendship is not inherited
*/

/*
class Node {
    int key;
    Node* next;

    // Class LinkedList can access private members of Node
    friend class LinkedList;

    
    // Search() of LinkedList can access internal members
    friend int LinkedList::search();
    
}; */

#include <iostream> 
class A { 
private: 
    int a; 
  
public: 
    A() { a = 0; } 
    friend class B; // Friend Class 
}; 
  
class B { 
private: 
    int b; 
  
public: 
    void showA(A& x) 
    { 
        // Since B is friend of A, it can access 
        // private members of A 
        std::cout << "A::a=" << x.a; 
    } 
}; 
  
int main() 
{ 
    A a; 
    B b; 
    b.showA(a); 
    return 0; 
} 



