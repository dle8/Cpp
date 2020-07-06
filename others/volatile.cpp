/*
    - volatile specifier is used to indicate that an object can be modified by something external to the thread of control. (*)
    - volatile specifier tells the compiler not to optimize away apparently redundant read and writes (**)
*/

// (*)
volatile const long clock_register; // updated by the hardware clock

// (**)
auto t1 {clock_register};
// ... no use of clock_register here ...
auto t2 {clock_register};

// if clock_register is not volatile, the compiler would have been perfectly entitled to eliminate one of the reads and assume t1 == t2.