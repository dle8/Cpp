/*
    Use decltype to deduce type without defining an initialized variable. Then decltype(expr) can be used as the
    declared type of expr.
*/

#include <bits/stdc++.h>
using namespace std;

template<typename T>
class MatrixRef {
public:
    int c;
    T* row;

    MatrixRef(int c): c(c) {
        row = (T*)malloc(c * sizeof(T));
    }

    T& operator[](int i) {
        return row[i];
    }

};

template<typename T>
class Matrix {
    public:
    MatrixRef<T> *mat;
    int r, c;

    Matrix(int r, int c): r(r), c(c) {
        mat = (MatrixRef<T>)*malloc(r * sizeof(MatrixRef<T>));
        for (int i = 0; i < r; ++i) mat[i](c);
    }

    Matrix& operator[](const int &r) {
        return mat[r];
    }

    int rows() { return r; }

    int cols() { return c; }
};

/*
    An example of generic programming that adds two matrices and return the matrix, whose element type of the sum is the
    type of the sum of the elements.

    The reason of -> in function heading can be explain in: https://stackoverflow.com/questions/22514855/arrow-operator-in-function-heading
    (so it's only available for C++11, which therefore need -std=c++11 flag for compilation)
*/
 
template<typename T, typename U>
auto operator+(const Matrix<T> &a, const Matrix<U> &b) -> Matrix<decltype(T{} + U{})> {
    Matrix<decltype(T{} + U{})> res;
    for (int i = 0; i != a.rows(); ++i) {
        for (int j = 0; j != a.cols(); ++j) {
            res[i][j] += a[i][j] + b[i][j];
        }
    }
}

int main() {

    MatrixRef<int> r(3);
    cout << "Hello new knowledge";
    return 0;
}