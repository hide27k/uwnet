#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // TODO: 6.1 - iterate over the input and fill in the output with max values

    for (int i = 0; i < in.rows; i += 1) {
            int outcol = 0;
            for (int c = 0; c < l.channels; c++)
                for (int y = 0; y < l.height; y += l.stride)
                    for (int x = 0; x < l.width; x += l.stride) {
                        float max = -1.e8f;
                        for (int fy = - (l.size - 1) / 2; fy < -(l.size - 1) / 2 + l.size; fy++) {
                                for (int fx = - (l.size - 1) / 2; fx < -(l.size - 1) / 2 + l.size; fx++){
                                        float num = 0.0f;
                                        if (y + fy >= 0 && y + fy < l.height && x + fx < l.width && x + fx>= 0)
                                                num = in.data[i * in.cols + c * l.height * l.width
                                                        + (y + fy) * l.width + (x + fx)];
                                         max = max < num ? num : max;
                                }
                        }
                        out.data[i * out.cols + outcol] = max;
                        outcol++;
                    }
    }

    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
matrix backward_maxpool_layer(layer l, matrix dy)
{
    matrix in    = *l.x;
    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.

    for (int i = 0; i < in.rows; i += 1) {
            int outcol = 0;
            for (int c = 0; c < l.channels; c++)
                for (int y = 0; y < l.height; y += l.stride)
                    for (int x = 0; x < l.width; x += l.stride) {
                        float max = -1.e8f;
                        int dx_i = -1;
                        for (int fy = - (l.size - 1) / 2; fy < -(l.size - 1) / 2 + l.size; fy++) {
                                for (int fx = - (l.size - 1) / 2; fx < -(l.size - 1) / 2 + l.size; fx++){
                                        float num = 0.0f;
                                        if (y + fy >= 0 && y + fy < l.height && x + fx < l.width && x + fx>= 0) {
                                                int temp = i * in.cols + c * l.height * l.width
                                                        + (y + fy) * l.width + (x + fx);
                                                num = in.data[temp];
                                                dx_i = max < num ? temp : dx_i;
                                        }
                                         max = max < num ? num : max;
                                }
                        }
                        if (dx_i != -1)
                                dx.data[dx_i] += dy.data[i * dy.cols + outcol];
                        outcol++;
                    }
    }
    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay){}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

