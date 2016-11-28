test = read.csv("data/test.csv")
test = test/255
test = data.matrix(test)
test.imgs = test
rm(test)
ch = 3
sz = sqrt(ncol(test.imgs)/ch )
dim(test.imgs) = c(nrow(test.imgs), sz, sz, ch)
test.img.array = transposeAllImages(test.imgs)
d = dim(test.img.array)
ch = d[4]
h = d[3]
w = d[2]
no.images = d[1]
epsilon = 0.1
dim(test.img.array) = c(no.images, h*w*ch)
test.img.array = test.img.array-rowMeans(test.img.array)
print("compute sigma ----------")
sigma = test.img.array %*% t(test.img.array)/no.images
print("sigma computation complete")
s = svd(sigma)
print("calculation of svd complete")
#D = diag(s$d)
print("diagonal complete")
U = s$u
test.img.array = U%*%diag(1/sqrt(s$d+epsilon))%*%t(U)%*%test.img.array
test.img.array = t(test.img.array)
dim(test.img.array) = c(h,w,ch, no.images)

res_predicted = predict(model, test.img.array)
res_labels <- max.col(t(res_predicted)) - 1
df = data.frame(id =1:5000, class=res_labels)
write.csv(df, "res10.csv", col.names = T, row.names = F)

df$class=as.factor(df$class)
summary(df$class)
