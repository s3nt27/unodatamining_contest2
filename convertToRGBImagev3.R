### function name: convertToRGBImage
### input data file from train and test data
### output: images with pixel coordinates and rgb channel values
library(grid)
#library(imager)
#library(png)
require(mxnet)
convertToRGBImages = function(data, i) {
  
  img.labels = c("airplane", "automobile", "bird",
                 "cat", "deer", "dog", "frog",
                 "horse", "ship", "truck")
  
  data.label = data$y
  # find out rgb channel values
  data = subset(data, select=-c(y))
  data.r = data[, 1:1024]
  data.g = data[, (1:1024)+1024]
  data.b = data[, (1:1024)+2048]
  img.data = NA
  print("The image is of --------- ")
  print(img.labels[data.label[i]])
  # for (i in 1:nrow(data)){
  #   r = matrix(unlist(data.r[i, ]/255), nrow=32, byrow=T)
  #   g = matrix(unlist(data.g[i, ]/255), nrow=32, byrow=T)
  #   b = matrix(unlist(data.b[i, ]/255), nrow=32, byrow=T)
  #   im = rgb(r, g, b)
  #   img.data[i] = im
  #   print(i)
  # }

    r = matrix(unlist(data.r[i, ]/255), nrow=32, byrow=T)
    g = matrix(unlist(data.g[i, ]/255), nrow=32, byrow=T)
    b = matrix(unlist(data.b[i, ]/255), nrow=32, byrow=T)
    print(dim(r))
    #grd.img = rgb(r,g,b)
    #dim(grd.img) = dim(r)
    
    img.data = abind(r,g,b, along = 3)
    #grid.raster(img.data, interpolate=FALSE)
  return(img.data)
}




## this function converts the matrix to images
# so that we can extract features - bag of words,
# hog, etc. 
convertAllRGBImages = function(data) {
  ch = 3 # number of channels
  
  img.labels = c("airplane", "automobile", "bird",
                 "cat", "deer", "dog", "frog",
                 "horse", "ship", "truck")
  
  data.label = data$y
  # find out rgb channel values
  
  img.data = NA
  data = t(data[, -1])#/255
  img.data = array(unlist(data), dim=c(nrow(data),32,32,ch))
  
  #grid.raster(img.data, interpolate=FALSE)
  return(img.data)
}





### this is to prepare the data for
### CNN


convertTAllRGBImages = function(data) {
  ch = dim(data)
  ch = 3
  
  img.labels = c("airplane", "automobile", "bird",
                 "cat", "deer", "dog", "frog",
                 "horse", "ship", "truck")
  rData = unlist(data)
  dim(rData) = c(nrow(data), 32, 32, ch) # this is the 
  # format for the mxnet input
  # check following reference
  
  #https://www.apprendimentoautomatico.it/
   # image-recognition-tutorial-
  #python-mxnet-deep-convolutional/#easy-footnote-1
  return(rData)
}



saveInFolders = function(images, labels, isTrain=T){
  
  img.labels = c("airplane", "automobile", "bird",
                 "cat", "deer", "dog", "frog",
                 "horse", "ship", "truck")
  maindir = "C:\\Users\\developer\\Box Sync\\statlearning-8766\\contest 2"
  if (isTrain ==T) {
    maindir = "C:\\Users\\developer\\Box Sync\\statlearning-8766
    \\contest 2\\train.imgs"
    
  }
  else{
    maindir = "C:\\Users\\developer\\Box Sync\\statlearning-8766
    \\contest 2\\test.imgs"
  }
  setwd(maindir)
  for (i in 1:length(img.labels)){
    subdir = img.labels[i]
    dir.create(file.path(mainDir, subDir), showWarnings = FALSE)
  }
  
  
  for (i in (1:nrow(images))) {
    chdir = img.labels[labels[i]]
    d = paste(i, ".png")
    A = img.data[i, , ,]
    
    X = rgb(A[,,1]/255, A[,,2]/255, A[,,3]/255)
    dim(X) = c(32,32)
    writePNG(t(X), d)
  }
  #setwd(file.path(mainDir, subDir))
  
  
}



saveInDimFolders = function(data, isTrain=T){
  
  
  
  labels = data$y
  # find out rgb channel values
  data = subset(data, select=-c(y))
  data.r = data[, 1:1024]
  data.g = data[, (1:1024)+1024]
  data.b = data[, (1:1024)+2048]
  img.data = NA
  
  img.labels = c("airplane", "automobile", "bird",
                 "cat", "deer", "dog", "frog",
                 "horse", "ship", "truck")
  maindir = getwd()
  if (isTrain ==T) {
    maindir = file.path(maindir, "data/train.imgs")
    
  }
  else{
    maindir = file.path(maindir, "data/test.imgs")
  }
  setwd(maindir)
  for (i in 1:length(img.labels)){
    subdir = img.labels[i]
    dir.create(file.path(maindir, subdir), showWarnings = FALSE)
  }
  
  
  for (i in (1:nrow(data))) {
    chdir = img.labels[labels[i]]
    d = paste0(i, ".png")
    chdir = file.path(maindir, chdir)
    d = file.path(chdir, d)
    
    r = matrix(unlist(data.r[i, ]/255), nrow=32, byrow=T)
    g = matrix(unlist(data.g[i, ]/255), nrow=32, byrow=T)
    b = matrix(unlist(data.b[i, ]/255), nrow=32, byrow=T)
    img.data = abind(r,g,b, along=3)
    writePNG(img.data, d)
    
  }
  #setwd(file.path(mainDir, subDir))
  
  
}




### convert entire dataset to grayscale image matrix
## for initial test

## test with all three algorithms 
## to change the images into grayscale
## lightness, luminosity and average

grayScaleDataset = function(img){
  img.labels = c("airplane", "automobile", "bird",
                 "cat", "deer", "dog", "frog",
                 "horse", "ship", "truck")
  
  proc = "luminosity"
  ## channels
  r = img[,,1]
  g = img[,,2]
  b = img[,,3]
  gray.imgs = NA
  if (proc == "lightness") {
    gray.imgs = (max(r,g,b) +min(r,g,b))/2
  } else if(proc =="average"){
    gray.imgs = (r + g + b) / 3
  }
  else if(proc == "luminosity") {
    gray.imgs = 0.21 * r + 0.72 * g + 0.07 * b
  }
  
  return(gray.imgs)
}
grayAllImages = function(data){
  ch = dim(data)
  ch = 1
  require(parallel)
  no.cores = detectCores() - 1 # take max(cores) - 1
  #initiate clusters 
  
  n = dim(data)
  no.images = n[1]  # take the number of images in training
  print(n[1])
  # and test
  cl = makeCluster(no.cores)   
  
  rData = parApply(cl, data, 1, grayScaleDataset
  )
  
  stopCluster(cl)
  rData = t(rData)
  print(dim(rData))
  sz = dim(rData)
  sz = sqrt(sz[2]/ch)
  dim(rData) = c(no.images, sz, sz, ch)
  return(rData)
}








## if needed you can transpose all the image for better viewing


transposeImg = function(img){
  ch = dim(img)
  ch = ch[3]
  
  #dim(image) = c(32, 32, ch)
  ## find the image if it is rgb
  # therefore there would be 3 channels
  for (i in 1:ch){
    transposedImg = (img[,,i])
    #dim(transposedImg) = c(32,32)
    transposedImg = t(transposedImg)
    
    img[,,i] = transposedImg
  }
  
  la = matrix(1, nc=3, nr=3)
  la[2,2] = -8
  #img = filter2(img, la)
  
  return(img)
}


transposeAllImages = function(data){
  ch = dim(data)
  ch = ch[4]
  require(parallel)
  no.cores = detectCores() - 1 # take max(cores) - 1
  #initiate clusters 
  
  n = dim(data)
  no.images = n[1]  # take the number of images in training
  print(n[1])
  # and test
  cl = makeCluster(no.cores)   
  clusterExport(cl, "filter2")
  rData = parApply(cl, data, 1, transposeImg
  )
  
  stopCluster(cl)
  rData = t(rData)
  print(dim(rData))
  
  sz = dim(rData)
  sz = sqrt(sz[2]/3)
  dim(rData) = c(no.images, sz, sz, ch)
  return(rData)
}
# this function tries to provide images
# with zero mean unit variance
perImageWhitening = function(img){
  ch = dim(img)
  ch = ch[3]
  whitenedImage = NA
  #dim(image) = c(32, 32, ch)
  ## find the image if it is rgb
  # therefore there would be 3 channels
  for (i in 1:ch){
    
    ch.img = (img[,,i])
    noe = dim(ch.img)
    noe = noe[1] * noe[2]

    
    adjusted.sdev = max(sd(ch.img), 
                        1.0/sqrt(noe))
    img[,,i] = (ch.img-mean(ch.img))#/adjusted.sdev
    
  }
  
  ## whiten the image now
  
  whitenedImage = img
  #dim(whitenedImage) = c(ch, 32, 32)
  return (whitenedImage)
}


## this function to whiten all images 
## with a parallel pool
## this should be always used as last 
# preprocessing function
whitenAllImages = function(data){
  epsilon = 0.1
  #epsiolon = 1+(epsilon + sd(data))  # scale
  ch = dim(data)
  ch = ch[4]
  sz= ch[2]
  require(parallel)
  no.cores = detectCores() - 1 # take max(cores) - 1
  #initiate clusters 
  
  n = dim(data)
  no.images = n[1]  # take the number of images in training
  print(n[1])
  # and test
  cl = makeCluster(no.cores)   
  
  rData = parApply(cl, data, 1, perImageWhitening
           )
  stopCluster(cl)
  print(dim(rData))
  rData = t(rData) # tranpose
  ### =====================================================
  ## this is where we perform ZCA
  print("compute sigma ----------")
  sigma = rData %*% t(rData)/no.images
  print("sigma computation complete")
  s = svd(sigma)
  print("calculation of svd complete")
  #D = diag(s$d)
  print("diagonal complete")
  U = s$u
  rData = U%*%diag(1/sqrt(s$d+epsilon))%*%t(U)%*%rData
  #rData = s$u %*%diag(1/sqrt(D)+epsilon) %*%t(U) %*%rData
  
  ### =====================================================
  
  
  
  rData = t(rData)
  print(dim(rData))
  #rData = t(rData)
  
  sz = dim(rData)
  sz = sqrt(sz[1]/3)
  dim(rData) = c(sz, sz, ch, no.images)
  
  return(rData)
}

## resize image to original size
## algorithm -> spline


resizePixels = function(im, w, h) {
  
  # initial width/height
  
  d = dim(im)
  ch = d[3]
  w1 = d[1]
  h1 = d[2]
  # target width/height
  w2 = w
  h2 = h
  # function to resize an image 
  # im = input image, w.out = target width, h.out = target height
  # Bonus: this works with non-square image scaling.
  
 
  
  # Create empty matrix
  #im.out = matrix(rep(0,w2*h2), nrow =w2, ncol=h2 )
  im.out = matrix(h2, w2, ch)
  
  # Compute ratios -- final number of indices is n.out, spaced over range of 1:n.in
  w_ratio = w1/w2
  h_ratio = h1/h2
  
  # Do resizing -- select appropriate indices
  for (i in 1:ch){
   img = im[,,i]
   im.out[,, ch] <- im[ floor(w_ratio* 1:w2), 
                        floor(h_ratio* 1:h2)]
  }
  return(im.out)
  
}

## random zoom/crop on image

randomZoomCenterImage = function(img){
  d = dim(img)
  # height and width
  h = d[1] 
  w = d[2]
  ## centers
  cx= round(h/2)
  cy= round(w/2)
  # cropped heights and widths
  cropw = 24
  croph = 24
  # halfs
  cropwh = round(cropw/2)
  crophh = round(croph/2)
  
  x1 =  max(1, cx - cropwh)
  y1 =  max(1, cy - crophh)
  
  x2 = min(w, cx+cropwh) 
  y2 = min(w, cy+crophh)
  
  rImg = img[4:27, 4:27, ]
  
  #dim(rImg) = c(no.images, h, w, ch)
  #rImg = resize(rImg, h, w)
  return(rImg)
    
  
}



randomZoomCenterAllImage = function(data){
  ch = dim(data)
  ch = ch[4]
  require(parallel)
  no.cores = detectCores() - 1 # take max(cores) - 1
  #initiate clusters 
  
  n = dim(data)
  no.images = n[1]  # take the number of images in training
  print(n[1])
  # and test
  cl = makeCluster(no.cores)
  clusterExport(cl, "resize")
  
  rData = parApply(cl, data, 1, randomZoomCenterImage
  )
  
  stopCluster(cl)
  rData = t(rData)
  print("Get the the resized dimension")
  print(dim(rData))
  print(ch)
  sz = dim(rData)
  sz = sqrt(sz[2]/3)
  dim(rData) = c(no.images, sz, sz, ch)
  #dim(rData) = c(no.images, 32, 32, ch)
  return(rData)
}


## random zoom/crop on image

randomZoomImage = function(img){
  d = dim(img)
  # height and width
  h = d[1] 
  w = d[2]
  
  w1 = 24
  h1 = 24
  ## centers
  cx= round(h/2)
  cy= round(w/2)
  A = seq(1, 8)
  A = sample(A)
  x1 = A[1]
  y1 = A[2]
  
  x2 = x1 + (w1-1)
  y2 = y1 + (h1-1)
   
  
  rImg = img[x1:x2, y1:y2, ]
  
  #dim(rImg) = c(no.images, h, w, ch)
  #rImg = resize(rImg, h, w)
  return(rImg)
  
  
}


randomZoomAllImage = function(data){
  ch = dim(data)
  ch = ch[4]
  require(parallel)
  no.cores = detectCores() - 1 # take max(cores) - 1
  #initiate clusters 
  
  n = dim(data)
  no.images = n[1]  # take the number of images in training
  print(n[1])
  # and test
  cl = makeCluster(no.cores)
  clusterExport(cl, "resize")
  
  rData = parApply(cl, data, 1, randomZoomImage
  )
  
  stopCluster(cl)
  rData = t(rData)
  print("Get the the resized dimension")
  print(dim(rData))
  print(ch)
  sz = dim(rData)
  sz = sqrt(sz[2]/3)
  dim(rData) = c(no.images, sz, sz, ch)
  #dim(rData) = c(no.images, 32, 32, ch)
  return(rData)
}


## random flip image
randomFlipImage = function(img){
  seed = runif(1, 0, 1)
  choice = 0
  rImg = NA 
 
  # 
  if (seed >= 0.5){
    choice = 1
  }else {
    choice = 0
  }
  
  if (choice == 1) {
    rImg = transposeImg(img)
  }else{
    rImg = img
  }
  return(rImg)
}

randomFlipAllImage = function(data){
  ch = dim(data)
  ch = ch[4]
  require(parallel)
  no.cores = detectCores() - 1 # take max(cores) - 1
  #initiate clusters 
  
  n = dim(data)
  no.images = n[1]  # take the number of images in training
  print(n[1])
  # and test
  cl = makeCluster(no.cores)   
  clusterExport(cl, "transposeImg")
  rData = parApply(cl, data, 1, randomFlipImage
  )
  
  stopCluster(cl)
  rData = t(rData)
  print(dim(rData))
  sz = dim(rData)
  sz = sqrt(sz[2]/3)
  dim(rData) = c(no.images, sz, sz, ch)
  
  return(rData)
}


# random brightness

randomDistortImage = function(img){
  ch.distort = runif(1, -.25,.25)
  d =dim(img)
  ch = d[3]
  #dim(img) = c(1024, ch)
  
  img = img + ch.distort
  for (i in 1:ch){
    im = img[,,i]
    im[im<0] = 0
    im[im>1] = 1
    img[,,i] = im
  }
  return(img)
}

randomDistortAllImages = function(data){
  ch = dim(data)
  ch = ch[4]
  require(parallel)
  no.cores = detectCores() - 1 # take max(cores) - 1
  #initiate clusters 
  
  n = dim(data)
  no.images = n[1]  # take the number of images in training
  print(n[1])
  # and test
  cl = makeCluster(no.cores)   
  
  rData = parApply(cl, data, 1, randomDistortImage
  )
  
  stopCluster(cl)
  print(dim(rData))
  rData = t(rData)
  print(dim(rData))
  sz = dim(rData)
  sz = sqrt(sz[2]/3)
  dim(rData) = c(no.images, sz, sz, ch)
  
  return(rData)
}

## random contrast of images


randomContrastImage = function(img){
  seed = runif(1, 0, 1)
  choice = 0
  upper = 1.8
  lower = 0.2
  ch.contrast = 0
  # 
  if (seed >= 0.5){
    choice = 1
  }else {
    choice = 0
  }
  
  if (choice == 1) {
    ch.contrast = runif(1, 1, upper)
  }else{
    ch.contrast = runif(1, lower, 1)
  }
  
  d =dim(img)
  ch = d[3]
 # dim(img) = c(1024, ch)
  
  img = (img * ch.contrast)
  
  for (i in 1:ch){
    im = img[,,i]
    im[im<0] = 0
    im[im>1] = 1
    img[,,i] = im
  }
  return(img)
}

randomContrastAllImages = function(data){
  ch = dim(data)
  ch = ch[4]
  require(parallel)
  no.cores = detectCores() - 1 # take max(cores) - 1
  #initiate clusters 
  
  n = dim(data)
  no.images = n[1]  # take the number of images in training
  print(n[1])
  # and test
  cl = makeCluster(no.cores)   
  
  rData = parApply(cl, data, 1, randomContrastImage
  )
  
  stopCluster(cl)
  print(dim(rData))
  rData = t(rData)
  print(dim(rData))
  
  sz = dim(rData)
  sz = sqrt(sz[2]/3)
  dim(rData) = c(no.images, sz, sz, ch)
  return(rData)
}




LecnLCN = function(X, im.shape, th=.0001, radius=9, div=T){
  
}
gaussFilter = function(kernel_shape){
  x = array(rep(0, (kernel_shape[1]*kernel_shape[2]*kernel_shape[3]*
                      kernel_shape[4])), kernel_shape)
  gauss = function(x, y, sigma=2){
    Z = 2 * pi * sigma ^ 2
    return  (1/ Z * exp(-(x ^ 2 + y ^ 2) / (2 * sigma ^ 2)))
  }
  mid = floor(kernel_shape[4] / 2)
 
  
  return (x /sum(x))
  
}


