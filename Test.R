rm(list = ls())

mpath="/Your Working Directory/"
setwd(mpath)

Version=22

source(paste0("D",Version,"_Functions.R"))

Pre      <- paste0("_",rawToChar(as.raw(65+Version)),"R_")
nt       <- 2000
n       <- 2000   
p        <- 20
iter     <- 100

SSIZE=c(2000)

SD=1



    
set.seed(1)        
x = matrix(runif(p * n, -2, 2), n, p)        
xtest = matrix(runif(p * nt, -2, 2), nt, p)        
f = function(x) {    10*sin(3.14*x[,1]*x[,2]) + 20*(x[,3]-0.5)^2+10*x[,4]+5*x[,5]  }
ftrue = f(x)
ftest = f(xtest)

if (SD==0)
{
sigma =sd(ftrue)
} else  {
sigma =SD
}



y = ftrue + sigma * rnorm(n)
y_test = ftest + sigma * rnorm(nt)


cat("Iteration ",1, " Sigma ",sigma,"\n")







t1=proc.time()

fit1=DSXBART(numnodes=4,
             CPath=mpath,
             Prefix=Pre,
             X=x,
             Y=y,
             X_TEST=xtest,
             beta= 1.1,
             num_burn = 30, 
             num_save = 60, 
             verbose=FALSE,
             binary=FALSE,
             binaryOffset=NULL,
             n_min=2,
             p_catagory=0,
             p_continuous=p,
             max_depth=10,
             n_cutpoints=100,
             RepeatT=6,
             Max_catagory=10,
             num_tree=50,
             mtry=10,
             Return_Result=TRUE,
             domh=FALSE,
             depthconstrain=TRUE,
             delhigest=FALSE,
             ver=Version,
             widetype=2,
             try=10,
             MixRestart=FALSE,
             selectType=4,
             winnercontrol=0)
t2=proc.time()
t=t2-t1

rmse_r=rmse(ftest,fit1$ytest)
rmse_t = round(fit1$RT[2,2], 3)





    





