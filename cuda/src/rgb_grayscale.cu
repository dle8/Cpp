#include <bits/stdc++.h>
using namespace std;

#define CHANNELS 3

__global__ void colorConvert(unsigned char* grayImage, unsigned char* rgbImage, int width, int height) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < width && y < height) {
        int coor = y * width + x;
        int r = coor * CHANNELS;
        int g = coor * CHANNELS + 1;
        int b = coor * CHANNELS + 2;

        grayImage[coor] = 0.21 * rgbImage[r] + 0.71 * g + 0.07 * b;
    }
}

int main() {
    
    return 0;
}