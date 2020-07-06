/*
- constexpr objects are const and are initialized with values known during compilation.
- constexpr functions can produce compile-time results when called with arguments whose values are known during compilation
- constexpr objects and functions maybe used in a wider range of contexts than non-constexpr objects and functions
    - constexpr objects can be created in read-only memory
    - strict line between compiling and runtime begins to blur, and some computation traditionally done at runtime can migrate to compile time
- constexpr is a part of an object's or function's template. Using it depends on your willingness to make a long-term commitment to the
constraints it imposes on the objects and functions you apply it to

- constexpr functions are limited to taking and returning literal types (that can have values determined during compilation).
*/

#include <bits/stdc++.h>
using namespace std;

class Point {
    public:
        constexpr Point(double xVal = 0, double yVal = 0) noexcept: x(xVal), y(yVal) {}

        /*
            getters xValue and yValue can be constexpr, because if they're invoked on a Point object with a value known during compilation
            then the values of the data members x and y can be know during compilation.
        */
        constexpr double xValue() const noexcept {
            return x;
        }

        constexpr double yValue() const noexcept {
            return y;
        }

        /*
        // C++ 11
        void setX(double newX) noexcept{ x = newX; }
        void setY(double newY) noexcept{ y = newY; }
        */

        // C++ 14
        constexpr void setX(double newX) noexcept { x = newX; }
        constexpr void setY(double newY) noexcept { y = newY; }
    
    private:
        double x, y;
};

constexpr Point midpoint (const Point &p1, const Point &p2) noexcept {
    return { (p1.xValue() + p2.xValue()) / 2, (p1.yValue() + p2.yValue()) / 2};
}

int main() {

    constexpr Point p1(9.4, 27.7);
    constexpr Point p2(28.8, 5.3);

    constexpr auto mid = midpoint(p1, p2);
    return 0;
}