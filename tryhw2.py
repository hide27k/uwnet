from uwnet import *
def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 8, 3, 2),
            make_convolutional_layer(8, 8, 8, 16, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 16, 3, 2),
            make_convolutional_layer(4, 4, 16, 32, 3, 1),
            make_activation_layer(RELU),
            make_connected_layer(512, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

def conv_net_with_batch():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
            make_batchnorm_layer(8),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 8, 3, 2),
            make_convolutional_layer(8, 8, 8, 16, 3, 1),
            make_batchnorm_layer(16),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 16, 3, 2),
            make_convolutional_layer(4, 4, 16, 32, 3, 1),
            make_batchnorm_layer(32),
            make_activation_layer(RELU),
            make_connected_layer(512, 10),
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
rate = .05
momentum = .9
decay = .001

m = conv_net_with_batch()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# 7.6 Question: What do you notice about training the convnet with/without batch normalization? How does it affect convergence? How does it affect what magnitude of learning rate you can use? Write down any observations from your experiments:
# TODO: Your answer
# The model with batch normalization coverged faster than the one without batch normalization under the same hyper parameters. 
# In the final iteration, the loss of the model with batch normalization was around 1.28 and the loss of the defaul one was around 1.49.
# Additionally, the final accuracy of the model with batch normalization was better than the default one. 

# The default model returned a training accuracy of 0.405460000038147 and a test accuracy of 0.4027000069618225.
# The model with batch normalization returned a training accuracy of 0.5450999736785889 and a test accuracy of 0.536300003528595.

# After comparing two models with same parameters, we tested out different parameters of the model with batch normalization.
# If we increased the iteration into 5000, the accuracy was also increased around 10% but the difference between a training acuracy and a test accuracy got increased.
# It means that the model might start overfitting.
# The updated model returned a training accuracy of 0.668940007686615 and a test accuracy of 0.6243000030517578.
