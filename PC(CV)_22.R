rm(list = ls())



get_OXBART_params <- function(y) {
  XBART_params = list(num_trees = 30, # number of trees 
                      num_sweeps = 40, # number of sweeps (samples of the forest)
                      n_min = 5, # minimal node size
                      alpha = 0.95, # BART prior parameter 
                      beta = 1.25, # BART prior parameter
                      mtry = 10, # number of variables sampled in each split
                      burnin = 15,
                      no_split_penality = "Auto",
                      max_depth = 250,
                      num_cutpoints = 50) # burnin of MCMC sample
  XBART_params$tau = var(y) / XBART_params$num_trees # prior variance of mu (leaf parameter)
  return(XBART_params)
}





mpath="D:/Working/02-Reading/Paper/MyPaper/Accelerate Distributed SoftBART/BDSXBART/VA/"
mpath="/home/RH/dsxbart/"
setwd(mpath)

Version=22

source(paste0("D",Version,"_Functions.R"))

Pre      <- paste0("_",rawToChar(as.raw(65+Version)),"R_")
nt       <- 2000
p        <- 20
iter     <- 100

SNB=1000
SNS=1000


run_sbart  = FALSE 
run_dsbart = TRUE 
run_xbart  = FALSE 


##Sample size Part
##Sample List to produce different Result files
##if want ordinal result,just set SSIZE a vector of one element

SSIZE=c(2000)

##CV part
##To specify different level of CV parameter
##CV special Part I

cv_act=TRUE
CV_params <- function() 
{
  LL=list()
  #Customer
  LLCount=14
  #num_saves
  LL[[1]]=c(60,120)
  #TREE DEPTH
  LL[[2]]=c(10)   ##6
  #TREE NUMBER
  LL[[3]]=c(50)
  #Numbers of workers
  LL[[4]]=c(4)
  #beta
  LL[[5]]=c(1.1)  
  #domh
  LL[[6]]=c(FALSE) ##TRUE,FALSE  
  #mixture
  LL[[7]]=c(FALSE)  
  #constrain
  LL[[8]]=c(TRUE) ##TRUE,FALSE 
  #delhigest->Burn
  LL[[9]]=c(30) ##TRUE,FALSE 
  #type
  LL[[10]]=c(4) ##1,2,3,4    
  #RepeatTime
  LL[[11]]=c(5)     
  #TRY
  LL[[12]]=c(10) 
  #Wide Type
  LL[[13]]=c(2)   
  #Winner Constrol
  LL[[14]]=c(0)   
    
  LC=rep(0,LLCount)
  TC=1
  for (i in (1:LLCount))
  {
    LC[i]=length(LL[[i]])
    TC=TC*LC[i]
  }
  LI=rep(0,LLCount)
  CV_params = list(LL,LLCount,LC,TC,LI) 
  names(CV_params) <- c("LL", "LLCount", "LC","TC","LI")
  return(CV_params)
}

CVP<-CV_params()

##Want Just one result,Set ##CV_act=FALSE 
##make sure the first element is the default setting  
##Uncomment the below two items if want Just one result of col one option

##cv_act=FALSE
##CVP$TC=1


if (run_sbart)
{
require(SoftBart)
}

SD=1


for (iSample in SSIZE)
{ 
    result=as.data.frame(matrix(data = NA,nrow=2*(iter+CVP$LLCount),ncol=CVP$TC+2))
    FNAME=paste("S",iSample,"SD",SD,sep = "-")
    n=iSample

    for(istep in (1:iter))
    {


        set.seed(istep)        
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


        cat("Iteration ",istep, " Sigma ",sigma,"\n")


        if(run_dsbart)
        {



            for (cvv in (0:(CVP$TC-1)))
            {  
                Tcvv=cvv
                for(t1 in (1:CVP$LLCount))
                {
                    CVP$LI[t1]=(Tcvv%%CVP$LC[t1])+1
                    Tcvv=Tcvv%/%CVP$LC[t1]
                }
                

                for (t2 in (1:CVP$LLCount))
                {
                    result[t2,cvv+1]=CVP$LI[t2]
                }
                
                

                for (t2 in (1:CVP$LLCount))
                {
                    result[CVP$LLCount+t2,cvv+1]=CVP$LL[[t2]][CVP$LI[t2]]
                }  


                t1=proc.time()

                fit1=DSXBART(numnodes=CVP$LL[[4]][CVP$LI[4]],
                                CPath=mpath,
                                Prefix=Pre,
                                X=x,
                                Y=y,
                                X_TEST=xtest,
                                beta= CVP$LL[[5]][CVP$LI[5]],
                                num_burn = CVP$LL[[9]][CVP$LI[9]], 
                                num_save = CVP$LL[[1]][CVP$LI[1]], 
                                verbose=FALSE,
                                binary=FALSE,
                                binaryOffset=NULL,
                                n_min=2,
                                p_catagory=0,
                                p_continuous=p,
                                max_depth=CVP$LL[[2]][CVP$LI[2]],
                                n_cutpoints=100,
                                RepeatT=CVP$LL[[11]][CVP$LI[11]],
                                Max_catagory=10,
                                num_tree=CVP$LL[[3]][CVP$LI[3]],
                                mtry=10,
                                Return_Result=TRUE,
                                domh=CVP$LL[[6]][CVP$LI[6]],
                                depthconstrain=CVP$LL[[8]][CVP$LI[8]],
                                delhigest=FALSE,
                                ver=Version,
                                widetype=CVP$LL[[13]][CVP$LI[13]],
                                try=CVP$LL[[12]][CVP$LI[12]],
                                MixRestart=CVP$LL[[7]][CVP$LI[7]],
                                selectType=CVP$LL[[10]][CVP$LI[10]],
                                winnercontrol=CVP$LL[[14]][CVP$LI[14]] )
                t2=proc.time()
                t=t2-t1

                result[2*CVP$LLCount+istep,cvv+1]=rmse(ftest,fit1$ytest)
                
                result[2*CVP$LLCount+istep+iter,cvv+1] = round(fit1$RT[2,2], 3)

                cat("Iteration ",istep," Type ",cvv,  " SDBart Cost ",result[2*CVP$LLCount+istep+iter,cvv+1]," Seconds","\n")
                cat("TEST RMSE:",result[2*CVP$LLCount+istep,cvv+1],"\n") 

                system(paste("rm ",Pre,"*",sep=""))

            }    
        }


        if(run_sbart)
        {

            t1=proc.time()
            
            
            fit2 <- SoftBart::softbart(X = x, Y = y, X_test = xtest, 
                                    hypers = SoftBart::Hypers(x, y, num_tree = 50, temperature = 1),
                                    opts = SoftBart::Opts(num_burn = SNB, num_save = SNS, update_tau = TRUE))
            
            
            t2=proc.time()
            t=t2-t1

            result[2*CVP$LLCount+istep,CVP$TC+1]=rmse(fit2$y_hat_test_mean, ftest)
            result[2*CVP$LLCount+istep+iter,CVP$TC+1]=t[3][[1]]

            cat("Iteration ",istep,  "SBart Cost",result[2*CVP$LLCount+istep+iter,CVP$TC+1],"Seconds","\n")
            cat("TEST RMSE:",result[2*CVP$LLCount+istep,CVP$TC+1],"\n")
        }
        
        ################################################################
        if (run_xbart) 
        {
          params = get_OXBART_params(y)
          time = proc.time()
          fit4 = XBART::XBART(as.matrix(y), 
                              as.matrix(x), 
                              as.matrix(xtest), 
                              p_categorical = 0,
                              params$num_trees, 
                              params$num_sweeps, 
                              params$max_depth,
                              params$n_min, 
                              alpha = params$alpha, 
                              beta = params$beta, 
                              tau = params$tau, 
                              s = 1, 
                              kap = 1,
                              mtry = params$mtry, 
                              verbose = FALSE,
                              num_cutpoints = params$num_cutpoints, 
                              parallel = FALSE, 
                              random_seed = 100, 
                              no_split_penality = params$no_split_penality)
          
          
          fit4_mean = apply(fit4$yhats_test[, params$burnin:params$num_sweeps], 1, mean)
          time = proc.time() - time
          print(time)
          
          
          ####File Name Part5   
  
          
          result[2*CVP$LLCount+istep,CVP$TC+2]=rmse(fit4_mean,ftest)
          result[2*CVP$LLCount+istep+iter,CVP$TC+2] = round(time[3], 3)
          
          cat("Iteration ",istep,  "XBart Cost",result[2*CVP$LLCount+istep+iter,CVP$TC+2],"Seconds","\n")
          cat("TEST RMSE:",result[2*CVP$LLCount+istep,CVP$TC+2],"\n")
          
        }
        #XBART-orgin
        ################################################################
    }


    if(run_sbart)
    {
        FNAME=paste(FNAME,
        "S",
        SNB,
        SNS,
        sep = "-")
    }                    

    if (run_xbart) 
    {
      FNAME=paste(FNAME,
                  "X",
                  params$num_sweeps,
                  sep = "-")  
    }
    
    FNAME=paste(FNAME,"-V",Version,"-T-",Sys.time(),".csv",sep="")
    write.table(result,FNAME,row.names=FALSE,col.names=TRUE,sep=",")
  
}

  

  
