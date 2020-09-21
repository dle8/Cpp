/*
Polymorphism means having many form. Typically, polymorphism occurs when there's a hierachy of classes and they are
 elated by inheritance.
C++ polymorphiims means that a call to a member function will cause a different functino ot be executed depending on the
 type of object that
invokes the function.

Without virtual keyword, the function area will be set once by the compiler as the version defined in the base class.
 This is called static
resolution of the function, or static linkage, or early binding (area() function is set during the compilation of the
 program).

Virtual function is a function in a base class that is declared using the keyword virtual. Defining in a base class a
 virtual function, with another version in a derived class, signals to the compiler that static linkage is not wanted
 for this function. What is wanted is the selection of the funciton to be called at any given point in the program to be
 based on the kind of object for which it's called - dynamic linkage, or late binding.

*/

#include <iostream>
using namespace std;

class Shape {
    protected:
        int width, height;
    
    public:
        Shape(int a = 0, int b = 0) {
            width = a;
            height = b;
        }
        
        virtual int area() {
            cout << "Parent class area: " << '\n';
            return 0;
        }

        // virtual int area() = 0;  pure virtual function
};

class Rectangle: public Shape {
    public:
        Rectangle(int a = 0, int b = 0): Shape(a, b) {}

        int area() {
            cout << "Rectangle class area: " << endl;
            return width * height;
        }
};

class Triangle: public Shape {
    public:
        Triangle(int a = 0, int b = 0): Shape(a, b) {}

        int area() {
            cout << "Triangle class area: " << '\n';
            return (width * height) / 2;
        }  
};

int main() {
    Shape *shape;
    Rectangle rec(10, 7);
    Triangle tri(10, 5);

    shape = &rec;
    shape->area();

    shape = &tri;
    shape->area();

    return 0;
}