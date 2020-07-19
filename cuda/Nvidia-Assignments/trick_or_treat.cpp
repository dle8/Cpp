/*
    Use sliding down to find the largest sum subarray that ends at i-th position. Subtract the first element of the subarray and move the first position to the right
    until the subarray sum is smaller or equal to the maximum number of candy.

    Time complexity: O(n)
    Space complexity: O(n)
*/

#include <bits/stdc++.h>
using namespace std;

void find_max_subarray(int* pieces, int homes, int max_candy, int &max_collected_candy, int &range_left, int &range_right) {
    int cur_sum = 0, left = 0, cur_max = 0;
    pair<int, int> range = make_pair(-1, -1);
    for (int i = 0; i < homes; ++i) {
        cur_sum += pieces[i];
        while (left <= i && cur_sum > max_candy) {
            cur_sum -= pieces[left++];
        }
        if (cur_sum > cur_max) {
            cur_max = cur_sum;
            range = make_pair(left, i);
        }
    }
    max_collected_candy = cur_max;
    range_left = range.first + 1;
    range_right = range.second + 1;
}

int main() {
    freopen("input.txt", "r", stdin);

    int homes, max_candy;
    cin >> homes >> max_candy;
    int pieces[homes];
    for (int i = 0; i < homes; ++i) cin >> pieces[i];

    fclose(stdin);

    int max_collected_candy = -1, range_left = -1, range_right = -1;

    find_max_subarray(pieces, homes, max_candy, max_collected_candy, range_left, range_right);

    if (max_collected_candy != -1) {
        cout << "Start at home " << range_left << " and go to home " << range_right<< " getting " << max_collected_candy << " pieces of candy" << endl;
    } else cout << "Don't go here" << endl;
    
    return 0;
}