## read training data
## change working directory






#################\\\

## will change it to 200 num rounds and 
#entire dataset (for bench marking)
### use num_rounds=150 for best result

## gives >55% when using dropout=0.5
## with 70% of training set
## use batch_size = 50

## please save this one as benchmark
## using this one 
## we get 52% validation accuracy
### please recheck
## substract rowmeans acucracy - 52%
## if you use grayimages- accuracy 47%
## if you use no ZCA- accuracy 48.4%
## no mean subtraction accuracy- 49.66%

##############
setwd("C:\\Users\\developer\\Documents\\rfiles\\contest 2\\contest 2")
#library(h2o)
library(doParallel)
library(foreach)

library(mxnet)  # for CNN
require(EBImage)
## source convertToRGBImage file

source("code/convertToRGBImagev3.R")
##
# read training data
train = read.csv("data/train.csv")
v = train[, -1]/255
train[, -1] = v

# change between 0.3 and 0.7 for better results
trainindex = sample(1:nrow(train), nrow(train)*0.7)

#train = grayScaleForDataset(train)
data.train = train[trainindex, ] #train data
data.test = train[-trainindex, ] # test data


### preprocess test and train  data

data.train = data.matrix(data.train)
data.test = data.matrix(data.test)

## preprocess it for mxnet input
train.x = (data.train[, -1])
test.x = (data.test[, -1])
train.x.array = convertTAllRGBImages(train.x)
train.x.array = transposeAllImages(train.x.array)



#train.x.array = grayAllImages(train.x.array)
#########################################################
# use this section when you want to pass
# raw images

#train.x.array = whitenAllImages(train.x.array)
d = dim(train.x.array)
ch = d[4]
h = d[3]
w = d[2]
no.images = d[1]
epsilon = 0.1
dim(train.x.array) = c(no.images, h*w*ch)
#train.x.array = scale(train.x.array, center=T)
train.x.array = train.x.array-rowMeans(train.x.array)
#train.x.array = test.x.array/sd(train.x.array)

#dim(train.x.array) = c(no.images, h, w,ch)
#train.x.array = whitenAllImages(train.x.array)

print("compute sigma ----------")
sigma = train.x.array %*% t(train.x.array)/no.images
print("sigma computation complete")
s = svd(sigma)
print("calculation of svd complete")
#D = diag(s$d)
print("diagonal complete")
U = s$u
train.x.array = U%*%diag(1/sqrt(s$d+epsilon))%*%t(U)%*%train.x.array
train.x.array = t(train.x.array)
dim(train.x.array) = c(h,w,ch, no.images)


#train.x.array = whitenAllImages(train.x.array)
##########################################################
library(png)
im1 = train.x.array[,,, 208]
writePNG(im1, "i3.png")

#im = w.train.x.array[,,,208]

# comment out grayAllImages if you want to use
# rgb image as mxnet input
# remember the conv net is an LEnet
#train.x.array = grayAllImages(train.x.array)


test.x.array = convertTAllRGBImages(test.x)
test.x.array = transposeAllImages(test.x.array)
#test.x.array = randomZoomCenterAllImage(test.x.array)
#test.x.array = grayAllImages(test.x.array)
## =======================================================
## use this instead of image whitening

#test.x.array = whitenAllImages(test.x.array)
d = dim(test.x.array)
ch = d[4]
h = d[3]
w = d[2]
no.images = d[1]

dim(test.x.array) = c(no.images, h*w*ch)
test.x.array = test.x.array - rowMeans(test.x.array)
#test.x.array = test.x.array/sd(test.x.array)
#dim(test.x.array) = c(no.images, h, w,ch)
#test.x.array = whitenAllImages(test.x.array)
#test.x.array = scale(test.x.array, center=T)
print("compute sigma ----------")
sigma = test.x.array %*% t(test.x.array)/no.images
print("sigma computation complete")
s = svd(sigma)
print("calculation of svd complete")
#D = diag(s$d)
print("diagonal complete")
U = s$u
test.x.array = U%*%diag(1/sqrt(s$d+epsilon))%*%t(U)%*%test.x.array
test.x.array = t(test.x.array)
dim(test.x.array) = c(h,w,ch, no.images)
#############################################################
# comment out grayAllImages if you want to use
# rgb image as mxnet input
# remember the conv net is an LEnet
#test.x.array = grayAllImages(test.x.array)
#w.test.x.array = whitenAllImages(test.x.array)

train.y = (data.train[, 1])
test.y = (data.test[, 1])




#w.train.x.array = whitenAllImages(train.x.array)
#w.test.x.array = whitenAllImages(test.x.array)

### do some image whitening here over test
### and train array

##no need for nor

# Set up the symbolic model
#-------------------------------------------------------------------------------

data <- mx.symbol.Variable('data')
# 1st convolutional layer
conv_1 <- mx.symbol.Convolution(data = data, kernel = c(3, 3), num_filter = 20)
tanh_1 <- mx.symbol.Activation(data = conv_1, act_type = "tanh")
pool_1 <- mx.symbol.Pooling(data = tanh_1, pool_type = "max", kernel = c(3, 3), stride = c(2, 2))
lrn_1 = mx.symbol.LRN(data=pool_1, alpha=0.0001, beta=0.75, knorm=1, nsize=5)

# 2nd convolutional layer
conv_2 <- mx.symbol.Convolution(data = lrn_1, kernel = c(3, 3), num_filter = 50)
tanh_2 <- mx.symbol.Activation(data = conv_2, act_type = "tanh")
pool_2 <- mx.symbol.Pooling(data=tanh_2, pool_type = "max", kernel = c(3, 3), stride = c(2, 2))
#lrn_2 = mx.symbol.LRN(data=pool_2, alpha=0.0001, beta=0.75, knorm=1, nsize=5)



# 1st fully connected layer
flatten <- mx.symbol.Flatten(data = pool_2)
flatten = mx.symbol.Dropout(data=flatten, p=0.824)
fc_1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = 1500)
bn_2 <- mx.symbol.BatchNorm(fc_1,eps=0.001,momentum=0.9,fix.gamma=T,name="bn_2")
tanh_3 <- mx.symbol.Activation(data = bn_2, act_type = "tanh")

# 2nd fully connected layer


fc_2 =mx.symbol.FullyConnected(data = tanh_3, num_hidden = 100)


# Output. Softmax output since we'd like to get some probabilities.
NN_model <- mx.symbol.SoftmaxOutput(data = fc_2)

# Pre-training set up
#-------------------------------------------------------------------------------

# Set seed for reproducibility
mx.set.seed(100)

# Device used. CPU in my case.
devices <- mx.cpu()

# Training
#-------------------------------------------------------------------------------

# Train the model
# take number of rounds atleast 100 for rgb or gray scale image
model <- mx.model.FeedForward.create(NN_model,
                                     X = train.x.array,
                                     y = train.y,
                                     ctx = devices,
                                     num.round = 400,
                                     array.batch.size = 50,
                                     learning.rate = 0.01,
                                     momentum = 0.9,
                                     #wd = 0.000001,
                                     eval.metric = mx.metric.accuracy,
                                     epoch.end.callback = mx.callback.log.train.metric(50),
                                     batch.end.callback = mx.callback.log.train.metric(50))

# Testing
#-------------------------------------------------------------------------------

# Predict labels
predicted <- predict(model, test.x.array)
# Assign labels
predicted_labels <- max.col(t(predicted)) - 1



table(test.y, predicted_labels)
# Get accuracy
mean(test.y== predicted_labels)

#
