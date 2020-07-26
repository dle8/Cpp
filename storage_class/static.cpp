/*
    static function: indicates that it is not associated with a particular object.
    static local variable: allows the function to perserve information between calls without introducing a global variable that might be 
        accessed and corrupted by other functions (*)
    static member: a variable that is part of a class, yet is not part of an object of that class. There is exactly one copy of a static 
        member instead of one object of that class (like non-static members)
*/

#include <bits/stdc++.h>
using namespace std;

void f(int a) {
    while (a--) {
        static int n = 0;
        int x = 0;

        cout << "n == " << n++ << ", x == " << x++ << '\n';
    }
}

int main() {
    f(3);
}