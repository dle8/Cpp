/* 
    Debug using standard error stream - cerr. Instead of debugging by printing out in standard
    iostream, we generate a whole new stream of data called the error stream. The difference is: 
    if we use freopen to open up files or submit the program to an OJ, the program will not include
    the error stream.
*/

inline void dbg() { cerr << "x is " << x << "\n"; }

int main() {
    dbg(); 
    x = 5000; 
    dbg(); 
}
