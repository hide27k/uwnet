from uwnet import *

def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(32, 32, 8, 3, 2),
            make_convolutional_layer(16, 16, 8, 16, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 16, 3, 2),
            make_convolutional_layer(8, 8, 16, 32, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 32, 3, 2),
            make_convolutional_layer(4, 4, 32, 64, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(4, 4, 64, 3, 2),
            make_connected_layer(256, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)
def fc_net():
    l = [   make_connected_layer(3072, 312),
            make_activation_layer(RELU),
            make_connected_layer(312, 278),
            make_activation_layer(RELU),
            make_connected_layer(278, 154),
            make_activation_layer(RELU),
            make_connected_layer(154, 72),
            make_activation_layer(RELU),
            make_connected_layer(72, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .005

m = conv_net()
print("training conv_net ...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating conv_net model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

m = fc_net()
print("training fc_net...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating fc_net model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# Now, for the unmodified convnet architecture in tryhw1.py, calculate how many
# operations it takes to process one forward pass through the network. You 
# only need to worry about the matrix operations since everything else is 
# pretty small compared to them. Let's assume that we have fused multiply-adds
# so a matrix multiplication of a M x K matrix with a K x N matrix takes M*K*N 
# operations. How many operations does the convnet use during a forward pass?

# We can first check the dimensions of the matrices involved for each convolution.
# For the first convolution layer convolutional_layer(32, 32, 3, 8, 3, 1, LRELU), we have a weight matrix of size 8 x 27 and an input matrix of size
# 27 x 1024. The convolution is just a matrix multiplication of the
# weight matrix and input. Therefore, we have 8 x 27 x 1024 = 221184 operations.

# Similarly, the second convolution layer convolutional_layer(16, 16, 8, 16, 3, 1, LRELU) has a weight matrix of size 16 x 72 and 
# an input matrix of size 72 x 256. Therefore, 16 x 72 x 256 = 294912 operations. 

# Third convolution layer convolutional_layer(8, 8, 16, 32, 3, 1, LRELU) has a weight matrix of size 32 x 144 and an input matrix of size 144 x 64. 
# Therefore, 32 x 144 x 64 = 294912 operations.

# Final convolution layer convolutional_layer(4, 4, 32, 64, 3, 1, LRELU) has a weight matrix of size 64 x 288 and an input matrix of size 288 x 16.
# Therefore, 64 x 288 x 16 = 294912 operations.

# The fully connected layer connected_layer(256, 10) simply has 2560 operations.# Thus, overall there are 1108480 operations through the convolution layers and one fully connected layer.

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
# Conv net acheived higher accuracies of 71% training and 67% test accuracies. The fully connected network achieved 57% training and 50% test accuracies. 
# Although both models have similar number of paramters, conv net performed better.
# This is likely due to a conv net's ability to observe an image's spatial features before making any predictions whereas 
# a fully connected network would try to make a prediction without reasoning with any spatial features. 

