/*
An interface describes the behavior or capabilities of a C++ class without committing to a particular implementation of that class. The C++
interface are implemented using abstract classes (ABC). A class is made abstract by declaring at least one of its functions are PURE VIRTUAL function.
A pure virtual function is specified by placing "=0" in its declaration as in following example: (*)

Purpose of ab ABC is to provide an appropriate base class from which other classes can inherit. Abstract classes cannot be used to instantiate
objects and servers only as an interface. Subclasses need to implement each of the virtual functions.

Classes than can be used to instantiate objects are called concrete classes.
*/

#include <iostream>
 
using namespace std;


// (*)
class Box {
    public:
        virtual double getVolume() = 0;

    private:
        double length, breadth, height;
};

// (**) 
// Base class
template<class T>
class Shape {
    public:
      // pure virtual function providing interface framework.
    
        virtual T getArea() = 0;
        
        void setWidth(T w) {
            width = w;
        }

        void setHeight(T h) {
            height = h;
        }

        void setDiameter(T d) {
            diameter = d;
        }
   
    protected:
        T width;
        T height;
        T diameter;
};
 
// Derived classes
class Rectangle: public Shape<int> {
   public:
        int getArea() { 
            return (width * height); 
        }
};

class Triangle: public Shape<int> {
   public:
        int getArea() { 
            return (width * height)/2; 
        }
};

class Circle: public Shape<float> {
    public:
        float getArea() { 
            return (diameter / 2) * 3.14; 
        }
};

 
int main(void) {
    Rectangle Rect;
    Triangle Tri;
    Circle Ci;

    Rect.setWidth(5);
    Rect.setHeight(7);

    // Print the area of the object.
    cout << "Total Rectangle area: " << Rect.getArea() << endl;

    Tri.setWidth(5);
    Tri.setHeight(7);

    // Print the area of the object.
    cout << "Total Triangle area: " << Tri.getArea() << endl; 

    Ci.setDiameter(3);

    // Print the area of the object.
    cout << "Total Circle area: " << Ci.getArea() << endl; 

    return 0;
}