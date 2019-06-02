rm(list = ls())
#LOADING PACKAGES
library(e1071)
library(dplyr)
library(dummies)
library(data.table)
require(ada)
#######################################
#SAMPLING THE DATA, SELECTING VARIABLES
#######################################

pitches <- read.csv('sample.csv') # Load the data
set.seed(1858) # Set a seed to replicate result
pitches <- na.omit(pitches)  #Get rid of any NA's
pitches$pitch_type<-as.character(pitches$pitch_type)
table(pitches$pitch_type) #Look at frequency

#Select pitch types
pitches%>% filter(pitch_type=="CH" | pitch_type=="CU" |
                  pitch_type=="FC" | pitch_type=="FF" |
                  pitch_type=="FT" |
                  pitch_type=="KC" |
                  pitch_type=="SI" | pitch_type=="SL" )->pitches

table(pitches$pitch_type) # Check

# Create Dummy Variables for the Pitch Types
pitches<- cbind(pitches, dummy(factor(pitches$pitch_type), sep = "_"))

pitches <- subset(pitches, select=-c(y0))  #Get rid of id and y0 (always 50)
pitches$code<-factor(pitches$code) #Change to factor
pitches$pitch_type<-factor(pitches$pitch_type)
pitches$on_1b<-factor(pitches$on_1b)
pitches$on_2b<-factor(pitches$on_2b)
pitches$on_3b<-factor(pitches$on_3b)
pitches$zone<-factor(pitches$zone)
pitches$zone<-factor(pitches$zone)
pitches$ab_id<-factor(pitches$ab_id)

# Rename y and x
setnames(pitches, old=c("x","y"), new=c("xl", "yl"))

#Create Smaller Data Set
#n <- nrow(pitches)
#samp_ind<- sample(n, .0002*n)  #Randomly sample the data
pitches_s <- pitches #Create a smaller data set
#summary(pitches_s) #Descriptive stats

# Create test and train sets for all pitch types
n <- nrow(pitches_s)
train_ind <- sample(n,.8*n) 

#####################################
######## TRAINING SETS ##############
#####################################

#Create train set for FF
pitches_s%>% select(-c(pitch_type, pitches_CH,
                                   pitches_CU,
                                   pitches_FC, pitches_FT,
                                   pitches_KC,
                                   pitches_SI, pitches_SL))->pitches_s_FF

#Create train set for CH
pitches_s%>% select(-c(pitch_type, pitches_FF,
                       pitches_CU,
                       pitches_FC, pitches_FT,
                       pitches_KC,
                       pitches_SI, pitches_SL))->pitches_s_CH

#Create train set for CU
pitches_s%>% select(-c(pitch_type, pitches_CH,
                       pitches_FF,
                       pitches_FC, pitches_FT,
                       pitches_KC,
                       pitches_SI, pitches_SL))->pitches_s_CU


#Create train set for FC
pitches_s%>% select(-c(pitch_type, pitches_CH,
                       pitches_CU,
                       pitches_FF, pitches_FT,
                       pitches_KC,
                       pitches_SI, pitches_SL))->pitches_s_FC

#Create train set for FT
pitches_s%>% select(-c(pitch_type, pitches_CH,
                       pitches_CU,
                       pitches_FC, pitches_FF,
                       pitches_KC,
                       pitches_SI, pitches_SL))->pitches_s_FT


#Create train set for KC
pitches_s%>% select(-c(pitch_type, pitches_CH,
                       pitches_CU,
                       pitches_FC, pitches_FT,
                       pitches_FF,
                       pitches_SI, pitches_SL))->pitches_s_KC

#Create train set for SI
pitches_s%>% select(-c(pitch_type, pitches_CH,
                       pitches_CU,
                       pitches_FC, pitches_FT,
                       pitches_KC,
                       pitches_FF, pitches_SL))->pitches_s_SI

#Create train set for SL
pitches_s%>% select(-c(pitch_type, pitches_CH,
                       pitches_CU,
                       pitches_FC, pitches_FT,
                       pitches_KC,
                       pitches_SI, pitches_FF))->pitches_s_SL

####################################
######## SUPPORT VECTOR MACHINE ####
####################################

# Four Seam Fast Ball (FF)
model_ff <- svm(pitches_FF~., data = pitches_s_FF[train_ind,],
                kernel = 'radial',
                gamma = 0.1,cost=2,
                scale=TRUE,
                probabilities=TRUE)

tune.outff <- tune(svm, pitches_FF~., data=pitches_s_FF[train_ind,], 
                 kernel="radial", 
                 ranges=list(cost=c(0.1, 1, 2, 3),
                             gamma=c(0.05, 0.1, 0.2)))

FF=predict(tune.outff$best.model, newdata=pitches_s_FF[-train_ind,])
svm_ff_pred<- rep(0,nrow(pitches_s_FF[-train_ind,]))
svm_ff_pred[FF>.5] = 1
table(true=factor(pitches_s_FF[-train_ind,]$pitches_FF), 
      pred=svm_ff_pred)

# Change Up (CH)
model_CH <- svm(pitches_CH~., data = pitches_s_CH[train_ind,],
                kernel = 'radial',
                gamma = 0.1,cost=2,
                scale=TRUE,
                probabilities=TRUE)

tune.outch <- tune(svm, pitches_CH~., data=pitches_s_CH[train_ind,], 
                 kernel="radial", 
                 ranges=list(cost=c(0.1, 1, 2),
                             gamma=c(0.1, 0.5, 1)))

CH=predict(tune.outch$best.model, newdata=pitches_s_CH[-train_ind,])

svm_CH_pred<- rep(0,nrow(pitches_s_CH[-train_ind,]))
svm_CH_pred[CH>.5] = 1
table(true=factor(pitches_s_CH[-train_ind,]$pitches_CH), pred=svm_CH_pred)

# Curve (CU)
model_CU <- svm(pitches_CU~., data = pitches_s_CU[train_ind,],
                kernel = 'radial',
                gamma = 0.1,cost=2,
                scale=TRUE,
                probabilities=TRUE)

tune.outcu <- tune(svm, pitches_CU~., data=pitches_s_CU[train_ind,], 
                   kernel="radial", 
                   ranges=list(cost=c(0.1, 1, 2),
                               gamma=c(0.1, 0.5, 1)))

CU=predict(tune.outcu$best.model, newdata=pitches_s_CU[-train_ind,])

svm_CU_pred<- rep(0,nrow(pitches_s_CU[-train_ind,]))
svm_CU_pred[CH>.5] = 1
table(true=factor(pitches_s_CU[-train_ind,]$pitches_CU), pred=svm_CU_pred)


# Cutter (FC)
model_FC <- svm(pitches_FC~., data = pitches_s_FC[train_ind,],
                kernel = 'radial',
                gamma = 0.1,cost=2,
                scale=TRUE,
                probabilities=TRUE)

tune.outfc <- tune(svm, pitches_FC~., data=pitches_s_FC[train_ind,], 
                   kernel="radial", 
                   ranges=list(cost=c(0.1, 1, 2),
                               gamma=c(0.1, 0.5, 1)))

FC=predict(tune.outfc$best.model, newdata=pitches_s_FC[-train_ind,])

svm_FC_pred<- rep(0,nrow(pitches_s_FC[-train_ind,]))
svm_FC_pred[FC>.5] = 1
table(true=factor(pitches_s_FC[-train_ind,]$pitches_FC), pred=svm_FC_pred)

# Two Seam Fast Ball (FT)
model_FT <- svm(pitches_FT~., data = pitches_s_FT[train_ind,],
                kernel = 'radial',
                gamma = 0.1,cost=2,
                scale=TRUE,
                probabilities=TRUE)

tune.outft <- tune(svm, pitches_FT~., data=pitches_s_FT[train_ind,], 
                   kernel="radial", 
                   ranges=list(cost=c(0.1, 1, 2),
                               gamma=c(0.1, 0.5, 1)))

FT=predict(tune.outft$best.model, newdata=pitches_s_FT[-train_ind,])

svm_FT_pred<- rep(0,nrow(pitches_s_FT[-train_ind,]))
svm_FT_pred[FT>.5] = 1
table(true=factor(pitches_s_FT[-train_ind,]$pitches_FT), pred=svm_FT_pred)


# Knuckle Curve (KC)
model_KC <- svm(pitches_KC~., data = pitches_s_KC[train_ind,],
                kernel = 'radial',
                gamma = 0.1,cost=2,
                scale=TRUE,
                probabilities=TRUE)

tune.outkc <- tune(svm, pitches_KC~., data=pitches_s_KC[train_ind,], 
                   kernel="radial", 
                   ranges=list(cost=c(0.1, 1, 2),
                               gamma=c(0.1, 0.5, 1)))

KC=predict(tune.outkc$best.model, newdata=pitches_s_KC[-train_ind,])

svm_KC_pred<- rep(0,nrow(pitches_s_KC[-train_ind,]))
svm_KC_pred[KC>.5] = 1
table(true=factor(pitches_s_KC[-train_ind,]$pitches_KC), pred=svm_KC_pred)

# Sinker (SI)
model_SI <- svm(pitches_SI~., data = pitches_s_SI[train_ind,],
                kernel = 'radial',
                gamma = 0.1,cost=2,
                scale=TRUE,
                probabilities=TRUE)

tune.outsi <- tune(svm, pitches_SI~., data=pitches_s_SI[train_ind,], 
                   kernel="radial", 
                   ranges=list(cost=c(0.1, 1, 2),
                               gamma=c(0.1, 0.5, 1)))

SI=predict(tune.outsi$best.model, newdata=pitches_s_SI[-train_ind,])

svm_SI_pred<- rep(0,nrow(pitches_s_SI[-train_ind,]))
svm_SI_pred[FC>.5] = 1
table(true=factor(pitches_s_SI[-train_ind,]$pitches_SI), pred=svm_SI_pred)

# Slider (SL)
model_SL <- svm(pitches_SL~., data = pitches_s_SL[train_ind,],
                kernel = 'radial',
                gamma = 0.1,cost=2,
                scale=TRUE,
                probabilities=TRUE)

tune.outsl <- tune(svm, pitches_SL~., data=pitches_s_SL[train_ind,], 
                   kernel="radial", 
                   ranges=list(cost=c(0.1, 1, 2),
                               gamma=c(0.1, 0.5, 1)))

SL=predict(tune.outsl$best.model, newdata=pitches_s_SL[-train_ind,])

svm_SL_pred<- rep(0,nrow(pitches_s_SL[-train_ind,]))
svm_SL_pred[SL>.5] = 1
table(true=factor(pitches_s_SL[-train_ind,]$pitches_SL), pred=svm_SL_pred)
###############################
####### PREDICTIONS SVM #######
###############################

predictions_svm<-as.data.frame(cbind(FF,CH,CU,FC,FT,KC,SI,SL))
predictions_svm$pred_pitch<-colnames(predictions_svm)[apply(predictions_svm,1,which.max)]
table(true=factor(pitches_s$pitch_type[-train_ind]), pred=predictions_svm$pred_pitch)
write.csv(predictions_svm, file = "predictions_svm4000.csv")

################################
############ ADA BOOST #########
################################

# Four Seam Fast Ball (FF)

bm<- ada(formula=pitches_FF ~.,
         data=pitches_s_FF[train_ind,],
         iter=50,
         bag.frac=0.5,
         control=rpart.control(maxdepth=1,
                               cp=0.01))

FF <- predict(bm, newdata=pitches_s_FF[-train_ind,],type='prob') #Can't have variable named y
ada_ff_pred<- rep(0,nrow(pitches_s_FF[-train_ind,]))
ada_ff_pred[FF[,2]>.5] = 1
table(true=factor(pitches_s_FF[-train_ind,]$pitches_FF), 
      pred=ada_ff_pred)

# Change Up (CH)
bm<- ada(formula=pitches_CH ~.,
         data=pitches_s_CH[train_ind,],
         iter=50,
         bag.frac=0.5,
         control=rpart.control(maxdepth=1,
                               cp=0.01))

CH <- predict(bm, newdata=pitches_s_CH[-train_ind,],type='prob') 
ada_ch_pred<- rep(0,nrow(pitches_s_CH[-train_ind,]))
ada_ch_pred[CH[,2]>.5] = 1
table(true=factor(pitches_s_CH[-train_ind,]$pitches_CH), pred=ada_ch_pred)

# Change Up (CU)
bm<- ada(formula=pitches_CU ~.,
         data=pitches_s_CU[train_ind,],
         iter=50,
         bag.frac=0.5,
         control=rpart.control(maxdepth=1,
                               cp=0.01))

CU <- predict(bm, newdata=pitches_s_CU[-train_ind,],type='prob') 
ada_cu_pred<- rep(0,nrow(pitches_s_CU[-train_ind,]))
ada_cu_pred[CU[,2]>.5] = 1
table(true=factor(pitches_s_CU[-train_ind,]$pitches_CU), pred=ada_cu_pred)


# Cutter (FC)
bm<- ada(formula=pitches_FC ~.,
         data=pitches_s_FC[train_ind,],
         iter=50,
         bag.frac=0.5,
         control=rpart.control(maxdepth=1,
                               cp=0.01))

FC <- predict(bm, newdata=pitches_s_FC[-train_ind,],type='prob') 
ada_fc_pred<- rep(0,nrow(pitches_s_FC[-train_ind,]))
ada_fc_pred[FC[,2]>.5] = 1
table(true=factor(pitches_s_FC[-train_ind,]$pitches_FC), pred=ada_fc_pred)

# Two Seam Fast Ball (FT)
bm<- ada(formula=pitches_FT ~.,
         data=pitches_s_FT[train_ind,],
         iter=50,
         bag.frac=0.5,
         control=rpart.control(maxdepth=1,
                               cp=0.01))

FT <- predict(bm, newdata=pitches_s_FT[-train_ind,],type='prob') 
ada_ft_pred<- rep(0,nrow(pitches_s_FT[-train_ind,]))
ada_ft_pred[FT[,2]>.5] = 1
table(true=factor(pitches_s_FT[-train_ind,]$pitches_FT), pred=ada_ft_pred)

# Knuckle Curve (KC)
bm<- ada(formula=pitches_KC ~.,
         data=pitches_s_KC[train_ind,],
         iter=50,
         bag.frac=0.5,
         control=rpart.control(maxdepth=1,
                               cp=0.01))

KC <- predict(bm, newdata=pitches_s_KC[-train_ind,],type='prob') 
ada_kc_pred<- rep(0,nrow(pitches_s_KC[-train_ind,]))
ada_kc_pred[KC[,2]>.5] = 1
table(true=factor(pitches_s_KC[-train_ind,]$pitches_KC), pred=ada_kc_pred)

# Sinker (SI)
bm<- ada(formula=pitches_SI ~.,
         data=pitches_s_SI[train_ind,],
         iter=50,
         bag.frac=0.5,
         control=rpart.control(maxdepth=1,
                               cp=0.01))

SI <- predict(bm, newdata=pitches_s_SI[-train_ind,],type='prob') 
ada_si_pred<- rep(0,nrow(pitches_s_SI[-train_ind,]))
ada_si_pred[SI[,2]>.5] = 1
table(true=factor(pitches_s_SI[-train_ind,]$pitches_SI), pred=ada_si_pred)

# Slider (SL)
bm<- ada(formula=pitches_SL ~.,
         data=pitches_s_SL[train_ind,],
         iter=50,
         bag.frac=0.5,
         control=rpart.control(maxdepth=1,
                               cp=0.01))

SL <- predict(bm, newdata=pitches_s_SL[-train_ind,],type='prob') 
ada_sl_pred<- rep(0,nrow(pitches_s_SL[-train_ind,]))
ada_sl_pred[SL[,2]>.5] = 1
table(true=factor(pitches_s_SL[-train_ind,]$pitches_SL), pred=ada_sl_pred)

###############################
####### PREDICTIONS ADA #######
###############################

predictions_ada<-as.data.frame(cbind(FF[,2],CH[,2],CU[,2],FC[,2],FT[,2],KC[,2],SI[,2],SL[,2]))
setnames(predictions_ada, old=c("V1","V2","V3","V4","V5","V6","V7","V8"), new=c("FF", "CH","CU","FC","FT","KC","SI","SL"))
predictions_ada$pred_pitch<-colnames(predictions_ada)[apply(predictions_ada,1,which.max)]
table(true=factor(pitches_s$pitch_type[-train_ind]), pred=predictions_ada$pred_pitch)
write.csv(predictions_ada, file = "predictions_ada4000.csv")
