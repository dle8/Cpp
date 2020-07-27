#include <bits/stdc++.h>
using namespace std;

int main() {
    unordered_map<string, string> umap( {{"apple","red"},{"lemon","yellow"}} );
    unordered_map<string, string> umap2 (second);  
    unordered_map<string, string> umap3(umap2.begin(), umap2.end());

    // operator []
    cout << umap["apple"] << '\n'; // O(1): average constant size. Worst case O(n)

    // find
    auto iter = umap.find("apple"); // O(1): average constant. Worst case O(n)

    // erase
    umap.erase("apple"); // O(1). erase(key)
    umap.erase(umap.begin()); // O(1). erase(position)
    umap.erase(umap.find("apple"), umap.end()); // O(n): erase by range

    // hash_function(): return the hash function object used by unodered_map container
    unordered_map<string, string>::hasher fn = umap.hash_function();
    cout << "Hash of \" this\" is " << fn(this) << '\n';

    // equal range: average case constant. worst case: linear in container size

    // iterate
    for (auto &x: umap) cout << x.first << " " << x.second;
    return 0;
}