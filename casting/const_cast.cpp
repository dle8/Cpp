/*
const_cast for getting write access to something declared const. It converts between types that
differ only in const and volatile qualifiers

static_cast: for reversing a well-defined implicit conversion. It converts between related types
such as one pointer type to another in the same class hierarchy, an integral type to an enumeration,
or a floating-point type to an integral type. It also does conversions defined by constructors and
conversion operators.

reinterpret_cast: for changing the meaning of bit patterns. It handles conversions between unrelated
types such as an integer to a pointer or a pointer to an unrelated pointer type.

dynamic_cast: for dynamically checked class hierarchy navigation. It does run-time checked 
conversion of pointers and references into a class hierarchy.
*/
