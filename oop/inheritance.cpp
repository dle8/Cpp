/*
    Multiple Inheritance is a feature of C++ where a class can inherit from more than one classes.
    The constructors of inherited classes are called in the same order in which they are inherited.

    In here we have diamond problem, which occurs when two superclasses of a class have a common base class.
    Use base class as virtual base. The language ensures that a constructor of a virtual base is called exactly one.
    the constructor of every virtual base is invoked (implicitly or explicitly) from the constructor of the most
    derived class to make sure constructor of base is called before its derived classes. If there are multiple constructor
    (parameterized ones and default), parameterized constructors can only be called within most derived class (****).

    Overriding virtual base functions: If different derived classes override the same function? This is only allow iff
    one function of a derived class override those of the others. If two classes override a base class function,
    but neither override the other, the class hierarchy is an error (ambiguous). That's why class TA need function show
    (***) else the compiler will report error.
*/

#include<iostream> 
using namespace std; 
class Person { 
public: 
    // parameterized constructor
    Person(int x)  { cout << "Person::Person(int ) called" << endl;   } 

    // default constructor
    Person()     { cout << "Person::Person() called" << endl;   } 

    virtual void show() {}
}; 
  
class Faculty : virtual public Person { 
public: 

    Faculty(int x):Person(x)   { 
       cout<<"Faculty::Faculty(int ) called"<< endl; 
    }

    Faculty():Person() {
        cout << "Faculty::Faculty() called" << endl;
    } 

    void show(int x) {
        cout << "faculty show " << x << '\n';
    }
}; 
  
class Student : virtual public Person { 
public: 

    Student(int x):Person(x) { 
        cout<<"Student::Student(int ) called"<< endl; 
    } 

    void show(string s) {
        cout << "student show " << s << '\n';
    }
}; 
  
class TA : public Faculty, public Student  { 
public: 

    using Student::show;

    TA(int x):Student(x), Faculty(x), Person(x)  { // ****
        cout<<"TA::TA(int ) called"<< endl; 
    } 

    // ***
    void show(int x) {
        cout << "ok";
    }
}; 
  
int main()  { 
    TA ta1(30); 
    ta1.show(3);
} 