
# consider time points as unit instead of date;
scaninitial0=function(tab1){
  n=dim(tab1)[1]
  init0times=NULL
  temp=NULL
  rowID=NULL
  slide=5 #5 pts constant increasing
  backward=2*30 # look back 60 pts
  forward=2*30
  for(i in (slide+forward-1):(n-backward)){
    if(all(diff(tab1[(i-slide+1):i,]$Close) >= 0)){
      temp=tab1[i,]
      init0times=rbind.data.frame(init0times,temp)
      rowID=rbind(rowID,i)
    }
  }
#   for(i in (slide+forward-1):n){
#     if(all(diff(tab1[(i-slide+1):i,]$Close) >= 0)){
#       temp=tab1[i,]
#       init0times=rbind.data.frame(init0times,temp)
#       rowID=rbind(rowID,i)
#     }
#  }
  return(list(row=rowID,table=init0times))
}

scanevent=function(tab1){
  backward=2*30 # look back 60 pts
  int0=tab1[1,]$Close
  tab1$closepct=tab1$Close/int0
  n=dim(tab1)[1]
  eventimes=NULL
  temp=NULL
  eventnum=NULL
  #define event within 60 pts
  for(i in 1:backward){
    if(tab1[i,]$closepct>=1.1)
    {temp=data.frame(Date=tab1[i,]$Date,Close=tab1[i,]$Close,Close_percent=tab1[i,]$closepct,Censor_type=1)
    eventnum=c(eventnum,i)
    eventimes=rbind.data.frame(eventimes,temp)}}  
  if(length(eventnum)==0) #if no events in 60 pts then give 60 days and censor type =0
  {temp=data.frame(Date=tab1[i,]$Date,Close=tab1[i,]$Close,Close_percent=tab1[i,]$closepct,Censor_type=0)
  eventimes=rbind.data.frame(eventimes,temp)
  daysto1st=60
  cens1st=0}else{
    daysto1st=eventnum[1]-1
    cens1st=1 
  }
  colnames(eventimes)=c("Date","Close","Close_percent","Censor_type")
  return(list(events=eventimes,day=daysto1st,cens=cens1st))
}


printevents=function(data,initialpts){
  ID=initialpts$row
  nrow=dim(data)[1]
  numevent=NULL
  for (j in 1:length(ID)){
    test=tail(data,n=nrow-ID[j]+1)
    res=scanevent(test)$events
    numevent=rbind(numevent,dim(res)[1])
  }
  print(table(numevent))
}

create_testset=function(data,initialpts,txtdir,imgdir){
  backward=2*30
  nrow=dim(data)[1]
  ID=initialpts$row
  r=NULL
  for (j in 1:length(ID)){
    test=tail(data,n=nrow-ID[j]+1)
    res=scanevent(test)
    tempres=data.frame(days=res$day,censortype=res$cens)
    write.table(tempres,paste(txtdir,"/imgs",j,".txt",sep = ""),sep = "\t",row.names = F,quote = F)
    imgtab=data[(ID[j]-backward+1):ID[j],]
    imgtab[,2:5]=imgtab[,2:5]/imgtab[backward,]$Close
    imgtab[,2:5]=log10(imgtab[,2:5]) #pct=price/init Close
    r1=range(imgtab[,2:5])
    r=c(r,r1[1],r1[2]) #log10(pct) to stablize y
    write.table(imgtab,paste(imgdir,"/imgs",j,".txt",sep = ""),sep = "\t",row.names = F,quote = F)
    }
  print(paste0("range of logpct:",min(r),"-",max(r)))
}


#module load intel/15.3
#module load mkl
#module load R/3.2.2
#R
# work in dragontooth

#setwd("/Users/manzhang/Documents/STAT PHD/Hong/Stock image prediction")
#install.packages('zoo')
#install.packages("lubridate")
#install.packages("tseries")
library(lubridate)
library(zoo)
library(tseries)

spComp <- read.table("SP500.txt",header = T,sep = "\t") 
## specify time period
dateStart <- "2007-04-01"               
dateEnd1 <- "2016-11-01"
dateEnd <- "2017-04-01"

## extract symbols and number of iterations
symbols <- as.character(spComp[, 1])
nAss <- length(symbols)
nAss = 10
## download data on first stock as zoo object
for (i in 138:nAss) {
  ## display progress by showing the current iteration step
  cat("Downloading ", i, " out of ", nAss , "\n")
  result <- try(z <- get.hist.quote(instrument = as.character(symbols[i]), start = dateStart,
                                    end = dateEnd, provider="yahoo",quote = c("Open", "High", "Low", "Close"),
                                    compression="d", retclass = "zoo", quiet = T))
  cmd=paste("mkdir -p testset_time/testset/ytab",i,sep = "")
  t1 <- try(system(cmd, intern = TRUE))
  cmd=paste("mkdir -p testset_time/testset/imgtab",i,sep = "")
  t1 <- try(system(cmd, intern = TRUE))
  cmd=paste("mkdir -p testset_time/trainset/ytab",i,sep = "")
  t1 <- try(system(cmd, intern = TRUE))
  cmd=paste("mkdir -p testset_time/trainset/imgtab",i,sep = "")
  t1 <- try(system(cmd, intern = TRUE))

  if(class(result) == "try-error") {
    next
  }
  else {
    tab=data.frame(z)
    tab$Date=rownames(tab)
    tab=tab[,c(5,1:4)]
    tab_sorted <- tab[order(tab$Date),] 
    tab_sorted$Date= strptime(tab_sorted$Date,format="%Y-%m-%d")
    whichend1=which(tab_sorted$Date==dateEnd1)
    train_tab=tab_sorted[1:whichend1,]
    test_tab=tab_sorted[(whichend1+1-60):(dim(tab)[1]),]
    train_init=scaninitial0(train_tab) #scan for # of start times
    test_init=scaninitial0(test_tab) #scan for # of start times
    if(!is.null(train_init$table)){
    dir1=paste("testset_time/trainset/ytab",i,sep = "")
    dir2=paste("testset_time/trainset/imgtab",i,sep = "")
    create_testset(train_tab,train_init,dir1,dir2)# create testset folder with imgs matching events
    print(paste(i,"th stock",sep = " "))
    }
    if(!is.null(test_init$table)){
    dir1=paste("testset_time/testset/ytab",i,sep = "")
    dir2=paste("testset_time/testset/imgtab",i,sep = "")
    create_testset(test_tab,test_init,dir1,dir2)# create testset folder with imgs matching events
    print(paste(i,"th stock test",sep = " "))
    }
  }
}



