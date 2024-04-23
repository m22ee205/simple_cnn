#include <iostream>

#define NUM_ROWS 24
#define NUM_COLS 24
#define FILTER_SIZE 5

void gradientCalculation(int grad_next_layer[NUM_ROWS][NUM_COLS], int filters[FILTER_SIZE][FILTER_SIZE], int grad_current_layer[NUM_ROWS][NUM_COLS]) {
    #pragma HLS PIPELINE
    #pragma HLS ARRAY_PARTITION variable=grad_next_layer complete dim=2
    #pragma HLS ARRAY_PARTITION variable=filters complete dim=0
    #pragma HLS ARRAY_PARTITION variable=grad_current_layer complete dim=2

    for (int r = 0; r < NUM_ROWS; ++r) {
        #pragma HLS UNROLL
        for (int c = 0; c < NUM_COLS; ++c) {
            #pragma HLS UNROLL
            grad_current_layer[r][c] = 0;
            for (int i = 0; i < FILTER_SIZE; ++i) {
                #pragma HLS UNROLL
                for (int j = 0; j < FILTER_SIZE; ++j) {
                    #pragma HLS UNROLL
                    int row_index = r + i - FILTER_SIZE / 2;
                    int col_index = c + j - FILTER_SIZE / 2;
                    if (row_index >= 0 && row_index < NUM_ROWS && col_index >= 0 && col_index < NUM_COLS) {
                        grad_current_layer[r][c] += grad_next_layer[row_index][col_index] * filters[i][j];
                    }
                }
            }
        }
    }
}
