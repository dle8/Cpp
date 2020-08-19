/*
    typedef is now rather outdated (though still used by some) because it is more or less just an 
    annoying version of using with frustrating semantics, so we will not cover it here.

    using is a fascinating keyword, frequently used to simplify namespace prefixing when applicable.
*/

using namespace std;
using std::cout; // to replace std::cout with cout

// Want to use string but neither want to use std::string or string?
using std::string;
using str =  string;
// or
using str = std::string;

// Even more alias
using ll = long long;
using str = string;
using pii = pair<int, int>;
using pll = pair<ll, ll>;
using vi = vector<int>;