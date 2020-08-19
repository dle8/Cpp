#include <bits/stdc++.h>

#include <stdio.h>
void my_int_func(int x) {
    printf( "%d\n", x );
}

int main() {
    void (*foo)(int);
    /*
        foo is a pointer to a function takes 1 integer argument and returns void. It's as if you're 
        declaring a function called "*foo", which takes an int and returns void; now, if *foo is a 
        function, then foo must be a pointer to a function. (Similarly, a declaration like int *x 
        can be read as *x is an int, so x must be a pointer to an int.)

        The key to writing the declaration for a function pointer is that you're just writing out 
        the declaration of a function but with (*func_name) where you'd normally just put func_name.
    */
    /* the ampersand is actually optional */
    foo = &my_int_func;
    foo(2);
    void *(*foo)(int *);
    /*
        Here, the key is to read inside-out; notice that the innermost element of the expression is 
        *foo, and that otherwise it looks like a normal function declaration. *foo should refer to a 
        function that returns a void * and takes an int *. Consequently, foo is a pointer to just 
        such a function.
    */

    return 0;
}