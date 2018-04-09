setwd("/Users/manzhang/Documents/STAT PHD/Hong/Stock image prediction/testset/")
save_in1="/Users/manzhang/Documents/STAT PHD/Hong/Stock image prediction/testset/"

# Set width
w <- 80
# Set height
h <- 60
#out_file <- "origsize_images.csv"
#out_file <- "resized80_60_images.csv"
out_file <- "scaled80_60_images.csv"

# List images in path
images <- list.files("/Users/manzhang/Documents/STAT PHD/Hong/Stock image prediction/testset/greyscaled_images/")
# Set up df
df <- data.frame()
# Set label/response
dir="/Users/manzhang/Documents/STAT PHD/Hong/Stock image prediction/testset/ytab/"
resfile=list.files(dir)
label=NULL
for(i in 1:length(resfile)){
  temp=read.table(paste(dir,resfile[i],sep = ""),header = T,sep = "\t")
  label=c(label,temp$days)
}
# source("https://bioconductor.org/biocLite.R")
# biocLite("EBImage")
require(EBImage)
# Main loop. Loop over each image
for(i in 1:length(images))
{
  # Read image
  img <- readImage(paste("/Users/manzhang/Documents/STAT PHD/Hong/Stock image prediction/testset/greyscaled_images/",images[i],sep = ""))
  # Reshape 60x60 to wxh
  #img2=resize(img,w=400,h=200)
  img2=img
  # Get the image as a matrix
  img_matrix <- img2@.Data
  #t=scale(img_matrix, center = TRUE, scale = TRUE)
  #t[is.na(t)] <- 0   !! centering and scaling no grey scale difference
  # Coerce to a vector
  img_vector <- as.vector(t(img_matrix))
  #img_vector <- as.vector(t(t))
  # Add label/response
  vec <- c(label[i], img_vector)
  # Bind rows
  df <- rbind(df,vec)
  # Print status info
  # print(paste("Done ", i, sep = ""))
}

img_size=800*600
# Set names
names(df) <- c("y", paste("pixel", c(1:img_size)))
# Write out dataset
write.csv(df, paste(save_in1,out_file,sep = ""), row.names = FALSE)

####### Traint CNN 
setwd("/Users/manzhang/Documents/STAT PHD/Hong/Stock image prediction/testset/")
#df=read.csv("resized80_60_images.csv")
df=read.csv("scaled80_60_images.csv")

#install.packages("drat", repos="https://cran.rstudio.com")
#drat:::addRepo("dmlc")
#install.packages("mxnet")
library(mxnet)
df=df[df$y!=60,]
# Train test datasets
ids=sample(1:(dim(df)[1]),100)
train <- df[ids,]
test <- df[-ids,]

# Fix train and test datasets
train <- data.matrix(train)
train_x <- t(train[,-1])
train_y <- train[,1]
train_array <- train_x
dim(train_array) <- c(80, 60, 1, ncol(train_x))

test_x <- t(data.matrix(test))
test_x <- t(test[,-1])
test_y <- test[,1]
test_array <- test_x
dim(test_array) <- c(80, 60, 1, ncol(test_x))

# Model 1 CNN
data <- mx.symbol.Variable('data')
conv1 <- mx.symbol.Convolution(data= data, kernel = c(3,3),stride = c(2, 2), num_filter = 20)
relu1 <- mx.symbol.Activation(data= conv1, act_type = "relu")
pool1 <- mx.symbol.Pooling(data = relu1, kernel = c(3, 3),pool_type = "avg")
conv2 <- mx.symbol.Convolution(data = pool1, kernel = c(3,3), num_filter = 20)
relu2 <- mx.symbol.Activation(data = conv2, act_type = "relu")
pool2 <- mx.symbol.Pooling(data = relu2, kernel = c(3, 3), pool_type = "avg")
conv3 <- mx.symbol.Convolution(data = pool2, kernel = c(3,3), num_filter = 20)
relu3 <- mx.symbol.Activation(data = conv3, act_type = "relu")
pool3 <- mx.symbol.Pooling(data = relu3, kernel = c(3, 3), pool_type = "avg")
conv4 <- mx.symbol.Convolution(data = pool3, kernel = c(3,3), num_filter = 10)
relu4 <- mx.symbol.Activation(data = conv4, act_type = "relu")
pool4 <- mx.symbol.Pooling(data = relu4, kernel = c(2, 2),stride = c(2, 2), pool_type = "avg")
# 1st fully connected layer
flat <- mx.symbol.Flatten(data = pool4)
flc1 <- mx.symbol.FullyConnected(data = flat, num_hidden = 1)
# linear regression Output (can use mx.symbol.SoftmaxOutput() for K-class prediction)
NN_model <- mx.symbol.LinearRegressionOutput(data = flc1)
# Set seed for reproducibility
mx.set.seed(100)
# Device used. Sadly not the GPU :-(
device <- mx.cpu()
# Train on 36 samples
model <- mx.model.FeedForward.create(NN_model, X = train_array, y = train_y,
                                     ctx = device,
                                     num.round = 30,
                                     array.batch.size = 20,
                                     learning.rate = 0.005,
                                     momentum = 0.9,
                                     wd = 0.00001,
                                     eval.metric = mx.metric.rmse)
#[30] Train-rmse=14.1257892317486 before scaling
#[30] Train-rmse=14.0868967060803 with scaling

# predict image y
preds <- predict(model, test_array) 
# test.y =
sqrt(mean((preds-test_y)^2)) #rmse 15.903

################## Deep neural network
data <- mx.symbol.Variable('data')
conv1 <- mx.symbol.Convolution(data= data, kernel = c(3,3),stride = c(2, 2), num_filter = 20)
relu1 <- mx.symbol.Activation(data= conv1, act_type = "relu")
pool1 <- mx.symbol.Pooling(data = relu1, kernel = c(3, 3),pool_type = "avg")
conv2 <- mx.symbol.Convolution(data = pool1, kernel = c(3,3), num_filter = 20)
relu2 <- mx.symbol.Activation(data = conv2, act_type = "relu")
pool2 <- mx.symbol.Pooling(data = relu2, kernel = c(3, 3), pool_type = "avg")
conv3 <- mx.symbol.Convolution(data = pool2, kernel = c(3,3), num_filter = 20)
relu3 <- mx.symbol.Activation(data = conv3, act_type = "relu")
pool3 <- mx.symbol.Pooling(data = relu3, kernel = c(3, 3), pool_type = "avg")
conv4 <- mx.symbol.Convolution(data = pool3, kernel = c(3,3), num_filter = 10)
relu4 <- mx.symbol.Activation(data = conv4, act_type = "relu")
pool4 <- mx.symbol.Pooling(data = relu4, kernel = c(2, 2),stride = c(2, 2), pool_type = "avg")
# 1st fully connected layer
flat <- mx.symbol.Flatten(data = pool4)
flc1 <- mx.symbol.FullyConnected(data = flat, num_hidden = 200)
relu5 <- mx.symbol.Activation(data = flc1, act_type = "relu")
flc2 <- mx.symbol.FullyConnected(data = relu5, num_hidden = 100)
relu6 <- mx.symbol.Activation(data = flc2, act_type = "relu")
flc3 <- mx.symbol.FullyConnected(data = relu6, num_hidden = 1)
set.seed(100)
NN_model <- mx.symbol.LinearRegressionOutput(data = flc3)
model <- mx.model.FeedForward.create(NN_model, X = train_array, y = train_y,
                                     eval.data = list(data=test_array,label=test_y),
                                     ctx = device,optimizer = "rmsprop",
                                     num.round = 30,
                                     array.batch.size = 20,
                                     learning.rate = 0.005,
                                     wd = 0.00001,
                                     eval.metric = mx.metric.rmse)
#[30] Train-rmse=13.2183826808659
#[30] Validation-rmse=19.4447501673524
preds <- predict(model, test_array) 
sqrt(mean((preds-test_y)^2)) #rmse 19.44475

#optimizer : (default is 'sgd')
#eval.data – validation set used during the process
#################################
## Model 2: MLP
set.seed(100)
model =  mx.mlp(data =as.matrix(t(train[,-1])), label=as.numeric(t(train[,1])),hidden_node = 100,out_node=1,
                device=mx.cpu(), num.round=100, array.batch.size=20, activation = "relu", out_activation = "rmse",array.layout = "columnmajor",
                learning.rate=0.0005,eval.metric=mx.metric.rmse,optimizer = "rmsprop")
preds=predict(model, t(test[,-1]),array.layout = "columnmajor")
#[100] Train-rmse=0.862589406731879
t(test[,1])-preds
sqrt(mean((test[,1]-preds)^2)) #16.91731

plot(test[,1], preds,xlab="Observed y", ylab="Fitted y", main=paste0("MLP"))
abline(0,1,col="red")

## 
data <- mx.symbol.Variable('data')
flat <- mx.symbol.Flatten(data = data)
flc1 <- mx.symbol.FullyConnected(data = flat, num_hidden = 200)
relu <- mx.symbol.Activation(data = flc1, act_type = "relu")
flc2 <- mx.symbol.FullyConnected(data = relu, num_hidden = 100)
relu1 <- mx.symbol.Activation(data = flc2, act_type = "relu")
flc3 <- mx.symbol.FullyConnected(data = relu1, num_hidden = 1)
NN_model <- mx.symbol.LinearRegressionOutput(data = flc3)
set.seed(100)
model <- mx.model.FeedForward.create(NN_model, X = train_array, y = train_y,
                                     eval.data = list(data=test_array,label=test_y),
                                     ctx = device,optimizer = "rmsprop",
                                     num.round = 100,
                                     array.batch.size = 20,
                                     learning.rate = 0.005,
                                     wd = 0.00001,
                                     eval.metric = mx.metric.rmse)
# [100] Train-rmse=8.86526858824535
# [100] Validation-rmse=14.7599963035251
preds <- predict(model, test_array) 
sqrt(mean((preds-test_y)^2)) #rmse 14.76



#################################
# Model 3: Restricted Boltzman Machine
#install.packages("darch")
library(darch)
train=data.frame(train)
test=data.frame(test)
############### better for now
set.seed(100)
model <- darch(y ~ ., train,preProc.targets=T, layers=c(1,20,50,20,1),darch.batchSize=10,bp.learnRate=0.001,darch.isClass=F,
               darch.errorFunction = "rmseError",darch.unitFunction=linearUnit)
#Train set RMSE: 0.144
preds=predict(model, newdata = test)
sqrt(mean((preds-test[,1])^2))
# 18.35171

set.seed(100)
#with resilient backpropogation
model=darch(train[,-1], train[,1], rbm.learnRate = 0.001, darch.batchSize=10,darch.isClass = FALSE, layers = c(1,20,50,20,1),preProc.targets =T,rbm.errorFunction="rmseError",
            darch.errorFunction = "rmseError",darch.fineTuneFunction = "rpropagation",darch.unitFunction=linearUnit,normalizeWeights=T)
#Train set RMSE: 8.4
preds=predict(model, newdata = test)
abs(preds-test[,1])

#DBN: minimizeAutoencoder
model <- darch(y ~ ., train,preProc.targets=T, layers=c(1,20,50,20,1),darch.batchSize=10,bp.learnRate=0.001,darch.isClass=F,
               darch.errorFunction = "rmseError",darch.unitFunction=linearUnit,darch.fineTuneFunction = "minimizeAutoencoder")
#Error: non-conformable arrays

####################################


####### visualize the filters 

data <- mx.symbol.Variable('data')
# 1st convolutional layer 5x5 kernel and 20 filters.
conv_1 <- mx.symbol.Convolution(data= data, kernel = c(5,5), num_filter = 20)
convact_1 <- mx.symbol.Activation(data= conv_1, act_type = "tahn")
pool_1 <- mx.symbol.Pooling(data = convact_1, pool_type = "max", kernel = c(2,2), stride = c(2,2))
# 2nd convolutional layer 5x5 kernel and 50 filters.
conv_2 <- mx.symbol.Convolution(data = pool_1, kernel = c(5,5), num_filter = 50)
convact_2 <- mx.symbol.Activation(data = conv_2, act_type = "tahn")
pool_2 <- mx.symbol.Pooling(data = convact_2, pool_type = "max", kernel = c(2,2), stride = c(2,2))
# 1st fully connected layer
flat <- mx.symbol.Flatten(data = pool_2)
fcl_1 <- mx.symbol.FullyConnected(data = flat, num_hidden = 50)
tanh_3 <- mx.symbol.Activation(data = fcl_1, act_type = "tahn")
# 2nd fully connected layer
fcl_2 <- mx.symbol.FullyConnected(data = tanh_3, num_hidden = 1)
# linear regression Output
CNN_model <- mx.symbol.LinearRegressionOutput(data = fcl_2)

train <- read.csv("resized60_60_sliced_36images.csv")
n=dim(train)[1]
k=10
set.seed(100)
id=sample(1:n, size = k, replace = F)
trainingset <- train[id,]
testset <- train[-id,]
train <- data.matrix(trainingset)
train_x <- t(train[,-1])
train_y <- train[,1]
train_array <- train_x
dim(train_array) <- c(60, 60, 1, ncol(train_x))
test = data.matrix(testset)
test_x <- t(test[,-1])
test_y <- test[,1]
test_array <- test_x
dim(test_array) <- c(60, 60, 1, ncol(test_x))

model <- mx.model.FeedForward.create(CNN_model, X = train_array, y = train_y,eval.data = list(data=test_array,label=test_y),num.round = 50, 
                                     array.batch.size = 20,learning.rate = 0.0005,momentum = 0.9,
                                     wd = 0.00001, #wd is weight decay 
                                     eval.metric = mx.metric.rmse)

out <- mx.symbol.Group(c(conv_1,convact_1, pool_1, conv_2, convact_2, pool_2,fcl_1, CNN_model))
# Create an executor
executor <- mx.simple.bind(symbol=out, data=dim(test_array), ctx=mx.cpu())

# Update parameters
mx.exec.update.arg.arrays(executor, model$arg.params, match.name=TRUE)
mx.exec.update.aux.arrays(executor, model$aux.params, match.name=TRUE)
# Select data to use
mx.exec.update.arg.arrays(executor, list(data=mx.nd.array(test_array)), match.name=TRUE)
# Do a forward pass with the current parameters and data
mx.exec.forward(executor, is.train=FALSE)
names(executor$ref.outputs)
# [1] "convolution1_output"            "activation1_output"             "pooling0_output"               
# [4] "convolution2_output"            "activation2_output"             "pooling1_output"               
# [7] "fullyconnected0_output"         "linearregressionoutput0_output"

#Visualize the output
#plot the output of the first 16 filters
# Plot the filters of the 7th test example
par(mfrow=c(4,4), mar=c(0.1,0.1,0.1,0.1))
for (i in 1:16) {
  outputData <- as.array(executor$ref.outputs$convolution1_output)[,,i,7]
  image(outputData,
        xaxt='n', yaxt='n',
        col=gray(seq(1,0,-0.05)))
}


