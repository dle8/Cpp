/*
    shared_ptr is similar to unique_ptr except that shared_ptr is copied rather than moved. The shared_ptr for an object share ownership of an object and that object is destroyed then the last of its shared_ptr is destroyed.
*/

#include <bits/stdc++.h>

using namespace std;

void f(shared_ptr<fstream>) {

}

void g(shared_ptr<fstream>) {

}

void user(const string&name, ios_base::openmode mode) {
    shared_ptr<fstream> fp {new fstream(name, mode)};
    if (!*fp) {
        cout << "No file";
        exit(0);
    }

    f(fp);
    g(fp);
}

int main() {
    // static constexpr openmode app = app;
    string file = "file_name";
    // user(file, app);
}