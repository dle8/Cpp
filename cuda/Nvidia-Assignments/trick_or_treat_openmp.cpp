/*
    Divide the original array into NUM_THREAD subarrays, each managed by one thread. For each thread, use sliding down to find the farest right point for each designated
    left point in the subarray it manages so that sum of all elements between these two points is maxmimum and smaller than maximum candy threshold.
    
    Time complexity: O(n) (since threads executions are overlapped)
    Space complexity: O(n)
*/


#include <bits/stdc++.h>
// #include <chrono>
#include <omp.h>
#define NUM_THREAD 4
#define CACHE_LINE_SIZE 64 // Cache line size for Intel core I7

using namespace std;

// Add padding to avoid false sharing. Each padded_i variable occupies an entire cache line.
struct padded_i{
	int sum, left, right;
	char padding[CACHE_LINE_SIZE - 3 * sizeof(int)];
};

void find_max_subarray(int *pieces , int homes , int max_candy, int &max_collected_candy, int &range_left, int &range_right) {
	int num_threads = min(NUM_THREAD, homes);
	omp_set_num_threads(num_threads);
    padded_i max_sum[num_threads];

	int length = (homes + num_threads - 1) / num_threads;

	#pragma omp parallel
	{	
        // Get current thread index
		int thread_index = omp_get_thread_num();
        max_sum[thread_index].sum = -1;
        // Range managed by thread_index thread is [thread_index * length, (thread_index + 1) * length)
        int start_section = thread_index * length, end_section = min(start_section + length, homes);
        // Value right - 1 dictates the end of subarray with maximum sum that smaller than max_candy
        int right = start_section, cur_sum = 0;
        for (int left = start_section; left < end_section; ++left) {
            while (right < homes && cur_sum + pieces[right] <= max_candy) {
                cur_sum += pieces[right++];
                if (cur_sum > max_sum[thread_index].sum && right > left) {
                    max_sum[thread_index].sum = cur_sum;
                    max_sum[thread_index].left = left;
                    max_sum[thread_index].right = right - 1;

                    // Condition to exit early
                    if (max_sum[thread_index].sum == max_candy) {
                        break;
                    }
                }
            }
        
            // Condition to exit early
            if (max_sum[thread_index].sum == max_candy) {
                break;
            }
            cur_sum -= pieces[left];
        }
	}

	// Get largest sum from all threads
    int sum_res = -1, tid = -1;
    for (int i = 0; i < num_threads; ++i) {
        if (max_sum[i].sum > sum_res) {
            sum_res = max_sum[i].sum;
            tid = i;
        }
    }
    
    max_collected_candy = sum_res;
    range_left = max_sum[tid].left + 1;
    range_right = max_sum[tid].right + 1;
}


int main(int argc , char *argv[]){
	freopen("input.txt", "r", stdin);

	int homes, max_candy, max_collected_candy = -1, range_left = -1, range_right = -1;;
	cin >> homes >> max_candy;
	int pieces[homes];
	for (int i = 0; i < homes; ++i) {
        cin >> pieces[i];
        if (max_candy == 0 && pieces[i] == 0 && range_left == -1) {
            range_left = range_right = i + 1;
            max_collected_candy = 0;
        }
    }

    fclose(stdin);

    auto start = chrono::steady_clock::now();

    if (max_candy != 0) {
        find_max_subarray(pieces, homes, max_candy, max_collected_candy, range_left, range_right);
    }

    auto end = chrono::steady_clock::now();
    cout << chrono::duration_cast<chrono::microseconds>(end - start).count() << "ms" << '\n';

    if (max_collected_candy != -1) {
        cout << "Start at home " << range_left << " and go to home " << range_right<< " getting " << max_collected_candy << " pieces of candy" << endl;
    } else cout << "Don't go here" << endl;

	
    return 0;
}


