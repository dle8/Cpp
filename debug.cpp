/* 
    Debug using standard error stream - cerr. Instead of debugging by printing out in standard
    iostream, we generate a whole new stream of data called the error stream. The difference is: 
    if we use freopen to open up files or submit the program to an OJ, the program will not include
    the error stream.
*/

#include <bits/stdc++.h>
using namespace std;

int x;

inline void dbg() { cerr << "x is " << x << "\n"; }

int main() {
    dbg(); 
    x = 5000; 
    dbg(); 
}

/*
    Address sanitizer using the flags: -ggdb -fsanitize=address,undefined 
    The first flag generates a debug report (in dSYM file format) based on the line numbering of the
    program, while the second flag can then access the dSYM file at runtime and give meaningful 
    errors. It helps diagnose errors that prevent the run flow of the program, such as out of 
    bounds, exceptions, and segmentation faults, even indicating precise line numbers.
*/
