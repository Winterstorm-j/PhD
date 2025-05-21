################################################
#SHINY DATA PREP FUNCTION used to prepare data into files accessed by shiny app
#file_path is the path to the folder - we use: 
#file_path<-("G:\\Dropbox\\ReAct\\SHINY_DATA_PREP\\Data_Output\\lab_1_ESS") ##MainzESS

#file_path<-("G:\\Dropbox\\ReAct\\SHINY_DATA_PREP\\Data_Output\\lab_1_NGM") ##MainzESS
#AT=50
#ShinyDataPrep(file_path, AT)
ShinyDataPrep<-function(file_path, AT){
  setwd(file_path)  
# List of required packages
required_packages <- c("fitdistrplus", "stringr", "readxl", "writexl", "gridExtra", "ggplot2", "grid")

# Check if the package is installed, if not, install it
for (package in required_packages) {
  if (!requireNamespace(package, quietly = TRUE)) {
    install.packages(package)
    suppressMessages(library(package, character.only = TRUE))
  } else {
    suppressMessages(library(package, character.only = TRUE))
  }
}  
x_value<-AT  
y=seq(0.01,10, by=0.001)  # quant values to evaluate

#The excel file to read is always the same name as the folder it is in

# Extract folder name from file path
folder_name <- basename(file_path)

# Concatenate folder name with .xlsx extension to get the file name
file_name <- paste0(folder_name, ".xlsx")

# Read the Excel file
dataframes_list <- readxl::read_xlsx(file.path(file_path, file_name), sheet = "data", col_names = TRUE)



# Read the "data" spreadsheet into a list of data frames
#dataframes_list <- read_excel(file_path, sheet = "data", col_names = TRUE)
# Assuming "ID" is the name of the ID column
unique_ids <- unique(dataframes_list$ID)
# Assuming "ID" is the name of the ID column
unique_ids <- unique(dataframes_list$ID)

# Loop through the unique IDs and create separate data frames
for (ID in unique_ids) {
  assign(ID, dataframes_list[dataframes_list$ID == ID, ])
}

# Now you can access the data frames like Ex_1, Ex_2, Ex3_0, Ex3_1, Ex3_2, etc.
mydatE1<-Ex_1
mydatE2<-Ex_2
mydat3_0<-Ex_3.0
mydat3_1<-Ex_3.1
mydat3_2<-Ex_3.2
########################
########################
#From the "information" spreadsheet read Ev value and lab name.
#Now make a separate spreadsheet with all of the backgound info attached
#setwd("G:\\Dropbox\\ReAct\\ANALYSIS\\Programming and analysis\\Data\\Raw_Files\\Mainz")
#Gen_dat<-read_xlsx("file_name", sheet ="information")
Gen_dat<-read_xlsx(file.path(file_path, file_name), sheet ="information")

# Set the row names to NULL
rownames(Gen_dat) <- NULL

lab_name_orig <- Gen_dat$Value[Gen_dat$Variable == "Laboratory Name"][1]
Ev <- Gen_dat$Value[Gen_dat$Variable == "Extraction_ volume (Ev)"][1]
Ev<-as.numeric(Ev)

# Open a .png device for plotting and specify the file path

# Specify the filename
file_name <- "pr_plot_Ex1.png"

# Create the full file path
full_file_path <- file.path(file_path, file_name)

# Check if the file already exists and remove it if it does
if (file.exists(full_file_path)) {
  file.remove(full_file_path)
}

# Open a .png device for plotting and specify the full file path
png(file = full_file_path, width = 800, height = 600)

# Define the vector FirstLastSwitch
FirstLastSwitch <- character(2)

Expt<-1
FirstLastSwitch[1]<-"Ex_1"
FirstLastSwitch[2]<-"LastH"
LastH_1<-Main(mydatE1$LastH,y,"green",Expt,FirstLastSwitch)##note use green to initialise the plot
FirstLastSwitch[2]<-"Back"
Background_1<-Main(mydatE1$Background,y,"orange",Expt,FirstLastSwitch)
title("Experiment 1")

# Close the .png device
dev.off()

#last handler and first handler Pr calcs. Background Pr is from a separate plot.
#the following stores parameters
#Plot expt2 lognormal distributions

# Open a .png device for plotting and specify the file path

# Specify the filename
file_name <- "combined_pr_plot.png"

# Create the full file path
full_file_path <- file.path(file_path, file_name)

# Check if the file already exists and remove it if it does
if (file.exists(full_file_path)) {
  file.remove(full_file_path)
}

# Open a .png device for plotting and specify the full file path
png(file = full_file_path, width = 800, height = 600)

#windows()
#plot.new()
# Set up a 2x2 grid for four plots on one page
par(mfrow = c(2, 2))

#calculate LastH Pr with respect to y (the array of quants of kwown person for LastH and FirstH and Bac - AsT and SecT)
#these probabilities are calculated using the lognormal distribution and passed to the Main function, 
#where they are plotted and $outC1 is returned with the array of probabilties per y quant value
#Purpose is to produce combined probability plots at this stage for all experiments
Expt<-2
FirstLastSwitch[1]<-"Ex_2"
FirstLastSwitch[2]<-"LastH"
LastH<-Main(mydatE2$LastH,y,"green",Expt,FirstLastSwitch)##note use green to initialise the plot
FirstLastSwitch[2]<-"FirstH"
FirstH<-Main(mydatE2$FirstH,y,"black",Expt,FirstLastSwitch)

FirstLastSwitch[2]<-"Back"
Background<-Main(mydatE2$Background,y,"orange",Expt,FirstLastSwitch)
title("Experiment 2")
#Plot expt3 lognormal distributions########
Expt<-3
FirstLastSwitch[1]<-"Ex_3.0"
FirstLastSwitch[2]<-"LastH"
LastH_3.0<-Main(mydat3_0$LastH,y,"green",Expt,FirstLastSwitch)##note use green to initialise the plot
FirstLastSwitch[2]<-"FirstH"
FirstH_3.0<-Main(mydat3_0$FirstH,y,"black",Expt,FirstLastSwitch)

FirstLastSwitch[2]<-"Back"
Background_3.0<-Main(mydat3_0$Background,y,"orange",Expt,FirstLastSwitch)
title("Experiment 3: 0 hours")
#######
FirstLastSwitch[1]<-"Ex_3.1"
FirstLastSwitch[2]<-"LastH"
LastH_3.1<-Main(mydat3_1$LastH,y,"green",Expt,FirstLastSwitch)##note use green to initialise the plot
FirstLastSwitch[2]<-"FirstH"
FirstH_3.1<-Main(mydat3_1$FirstH,y,"black",Expt,FirstLastSwitch)

FirstLastSwitch[2]<-"Back"
Background_3.1<-Main(mydat3_1$Background,y,"orange",Expt,FirstLastSwitch)
title("Experiment 3: 1 hour")
########
FirstLastSwitch[1]<-"Ex_3.1"
FirstLastSwitch[2]<-"LastH"
LastH_3.2<-Main(mydat3_2$LastH,y,"green",Expt,FirstLastSwitch)##note use green to initialise the plot
FirstLastSwitch[2]<-"FirstH"
FirstH_3.2<-Main(mydat3_2$FirstH,y,"black",Expt,FirstLastSwitch)

FirstLastSwitch[2]<-"Back"
Background_3.2<-Main(mydat3_2$Background,y,"orange",Expt,FirstLastSwitch)
title("Experiment 3: 2 hours")

# Close the .png device
dev.off()
################################################
################################################

######CALCULATE LOWER QUANT THRESHOLD VALUE BASED ON RFU
###AND CALC THE RFU V QUANT REGR PLOT
#Need to calculate expected background level when none is seen. This can be done from calculating the limit of detection
#Choose highest RFU to be conservative if dyes are different
#with this example 50orfu=AT
# we multiply by 2 to accomodate 2 alleles per locus which gives
#Average RFU=100RFU (2 alleles) in candidate example and use expt 1 for data
#Store result in predicted_y = quant equiv to 100rfu ave
predicted_y<-CalcUthresh(mydatE1,x_value,file_path)##Also gives regression plot
#ThreshResults calculate Prs for predicted_y value - ie the quant eqivalent to the AT value
##Expt2 threshold results
ThreshResults<-ThreshCalc(Background,LastH,FirstH,predicted_y)
ThreshResults<-as.numeric(ThreshResults)
BackthreshPr<-ThreshResults[1]
LastthreshPr<-ThreshResults[2]
FirstthreshPr<-ThreshResults[3]
####
##Expt3 threshold results
ThreshResults_3.0<-as.numeric(ThreshCalc(Background_3.0,LastH_3.0,FirstH_3.0,predicted_y))
BackthreshPr_3.0<-ThreshResults_3.0[1]
LastthreshPr_3.0<-ThreshResults_3.0[2]
FirstthreshPr_3.0<-ThreshResults_3.0[3]
##
ThreshResults_3.1<-as.numeric(ThreshCalc(Background_3.1,LastH_3.1,FirstH_3.1,predicted_y))
BackthreshPr_3.1<-ThreshResults_3.1[1]
LastthreshPr_3.1<-ThreshResults_3.1[2]
FirstthreshPr_3.1<-ThreshResults_3.1[3]
##
ThreshResults_3.2<-as.numeric(ThreshCalc(Background_3.2,LastH_3.2,FirstH_3.2,predicted_y))
BackthreshPr_3.2<-ThreshResults_3.2[1]
LastthreshPr_3.2<-ThreshResults_3.2[2]
FirstthreshPr_3.2<-ThreshResults_3.2[3]

###Next calculate the LR for a whole range of quants and plot for condition where there is no unknown person
##For the demo we plot a series of LRs for various U values
#############
#########Step to carry out calcs. to determine probs using log normal distributions from "analysisscript2"


#BNresults<-mapply(BNprog,FirstH,LastH,Background,AstU=1,y)
#Function requires: BNformula<-function(SecT,AsT,Bac,AsTU,BackthreshPr,LastthreshPr)
#Values of Prs related to quants stored in y are in the outC1 vectors. Need to avoid y=0 because computation will fail
#and the Prs are calculated relative to the quants of first and last handlers that are recovered
#???For POI only calculations we use BackThreshPr and LastthreshPr values to calc 1-background and 1-last handler values under Hd
#????We keep these values as constants so that we can plot results in 2D graphs - but some work needed to find out the sensitivity
#EXPERIMENT 2

### POI ONLY recovered so that U quant must be 0
# Notes - the sequence of data inputs into the BNformulaExpt2 function is: function(SecT,AsT,Bac,AsTU) or s,t,b,t':
#where SecT is the quant of the first handler, AsT is the quant of the last handler, 
#Bac is the quant of the background. AsTU is the quant of the unknown
#With POI only there is no unknown so we need Pr Back and Pr ASTU (from lastH) where y=0  adjusted to y=0.01 to avoid log(0) errors
#LRresults<-mapply(BNformulaExpt2,FirstH$outC1,LastH$outC1,Background$outC1,LastH$outC1) # Calc LR results expt2
#We can use the y values already calculated in $outC1 for SecT, AsT. But we need to calculate a single value for unknowns which
#we do later.
#Start with POI only means that we do not see U DNA, and we apply Pr U for background and last handler ASTU as a single Value
#we need to extract these results from Background$outC1 and LastH$outC1 values already calculated

###############################
#Experiment 2
Expt=2##IMPORTANT SWITCH
stringtopass="Exp_2_plot.png"
excelfilename="Results_Exp_2"
# Check if the file already exists and remove it if it does
if (file.exists(stringtopass)) {
  file.remove(stringtopass)
}
P0=exp3_plot(FirstH,LastH,Background,LastH,BackthreshPr,LastthreshPr, FirstthreshPr,y,stringtopass,Expt,excelfilename)

###########################
Expt=3##IMPORTANT SWITCH
#Expt3 0h
stringtopass="Exp_3.0_plot.png"
excelfilename="Results_Exp_3.0"
# Check if the file already exists and remove it if it does
if (file.exists(stringtopass)) {
  file.remove(stringtopass)
}
P1=exp3_plot(FirstH_3.0,LastH_3.0,Background_3.0,LastH_3.0,BackthreshPr_3.0,LastthreshPr_3.0, FirstthreshPr_3.0,y,stringtopass,Expt,excelfilename)

##Expt3 1h
stringtopass="Exp_3.1_plot.png"
excelfilename="Results_Exp_3.1"
# Check if the file already exists and remove it if it does
if (file.exists(stringtopass)) {
  file.remove(stringtopass)
}

P2=exp3_plot(FirstH_3.1,LastH_3.1,Background_3.1,LastH_3.1,BackthreshPr_3.1,LastthreshPr_3.1, FirstthreshPr_3.1,y,stringtopass,Expt,excelfilename)

#Expt3 2h
stringtopass="Exp_3.2_plot.png"
excelfilename="Results_Exp_3.2"
# Check if the file already exists and remove it if it does
if (file.exists(stringtopass)) {
  file.remove(stringtopass)
}

P3=exp3_plot(FirstH_3.2,LastH_3.2,Background_3.2,LastH_3.2,BackthreshPr_3.2,LastthreshPr_3.2, FirstthreshPr_3.2,y,stringtopass,Expt,excelfilename)
}

####################################################################################################################################################################
####################################################################################################################################################################
################### STATISTIC_FUN
#statistic_function<-function(file_path,Expt,y_known_quant,y_unknown_quant,AT){#use this if filepath
statistic_function<-function(data_in,Expt,y_known_quant,y_unknown_quant,AT){#use this if file
  #This function calculates LRs from a set of data
#for test data is a file_path - if you want to accept a file direct this goes back into the React() function
# Read the Excel file into a list of data frames
# Read the "data" spreadsheet into a list of data frames
#Read the expt number as 1,2,3.0,3.1 or 3.2
##SWITCH###dataframes_list <- read_excel(file_path, sheet = "data", col_names = TRUE)###MOVE BACK INTO REACT()and REACTX1 IF FILENAME CALLE
#organise and return data
data<-React(data_in,Expt)#load data corresponding to Expt2,3...etc
mydatE1<-ReactEx1(data_in)#load data from Expt1 for use to calc predicted_y value (quant. ng) based on AT RFU

# Define the vector FirstLastSwitch
FirstLastSwitch <- character(2)
Expt <- as.numeric(Expt)
## Input the expt no. into FirstLastSwitch[1]
if (Expt == 1) {
  FirstLastSwitch[1] <- "Ex_1"
} else if (Expt == 2) {
  FirstLastSwitch[1] <- "Ex_2"
} else if (Expt == 3.0) {
  FirstLastSwitch[1] <- "Ex_3.0"
} else if (Expt == 3.1) {
  FirstLastSwitch[1] <- "Ex_3.1"
} else if (Expt == 3.2) {
  FirstLastSwitch[1] <- "Ex_3.2"
} else {
  # Throw an error message if no conditions are satisfied
  stop("No Expt specified. Program failed.")
}
#The Prs are recalculated as follows for expt2:
FirstLastSwitch[2]<-"LastH"
LastH<-MainShiny(data$LastH,FirstLastSwitch)##this routine calculates the log normal parameters for last, first and background handlers
FirstLastSwitch[2]<-"FirstH"
FirstH<-MainShiny(data$FirstH,FirstLastSwitch)

FirstLastSwitch[2]<-"Back"
Background<-MainShiny(data$Background,FirstLastSwitch)
# Place a threshold of 0.05 for y quant value when y quant value is 0 (POI only condition)
if (y_unknown_quant==0){y_unknown_quant<-0.01}

##Calculate Prs to go into the BN
y_SecT_Test<-LNdist(FirstH$meanlog,FirstH$sdlog,FirstH$k,y_known_quant,colour="blue")## these probabilties feed into the BN
y_AsT_Test<-LNdist(LastH$meanlog,LastH$sdlog,LastH$k,y_known_quant,colour="blue")
y_Bac_Test<-LNdist(Background$meanlog,Background$sdlog,Background$k,y_unknown_quant,colour="blue")
y_AsTU_Test<-LNdist(LastH$meanlog,LastH$sdlog,LastH$k,y_unknown_quant,colour="blue")
###If y values are =1 then this resultsin infinity and the program crashes - fixed with CalclnCnew
#if (y_SecT_Test==1){y_SecT_Test<-0.99}
#if (y_AsT_Test==1){y_AsT_Test<-0.99}
#if (y_Bac_Test==1){y_Bac_Test<-0.99}
#if (y_AsTU_Test==1){y_AsTU_Test<-0.99}


##Now calculate threshold values so we can calculate Pr 1-b and 1-Ast - when no background or no unknown perpetrator observed
##where predicted y is the lower threshold quant value from 2 x AT value that is input and tested against the regression from expt1 
#predicted_y is calculated from the expt1 results for consistency across all expts using CalcUthresh()
#and file from expt 1 is always used: mydatE1
predicted_y<-CalcUthresh(mydatE1, AT,new_file_path=NULL)

##Expt2,3 threshold results the Pr of a result being higher than the threshold value for Background, LastH and FirstH respectively
ThreshResults<-ThreshCalc(Background,LastH,FirstH,predicted_y)
ThreshResults<-as.numeric(ThreshResults)
BackthreshPr<-ThreshResults[1]
LastthreshPr<-ThreshResults[2]
FirstthreshPr<-ThreshResults[3]
################

#Now calculate the LR_test results
#$LRPOI is the LR where there is no unknown
#$LRTog is the LR where there are unknowns

## Choose between BNformulaExpt2 or BNformulaExpt3 depending on the experiment
if (Expt==2){
LR_Test_results<-BNformulaExpt2(y_SecT_Test,y_AsT_Test,y_Bac_Test,y_AsTU_Test)
BNTresults<-BNformulaExpt2(FirstthreshPr,LastthreshPr,BackthreshPr,LastthreshPr)#Calc the threshold results into BNTresults
}else{
  LR_Test_results<-BNformulaExpt3(y_SecT_Test,y_AsT_Test,y_Bac_Test,y_AsTU_Test)
  BNTresults<-BNformulaExpt3(FirstthreshPr,LastthreshPr,BackthreshPr,LastthreshPr)#Calc the threshold results into BNTresults
}
return(list(LR_Test_results=LR_Test_results,BNTresults=BNTresults,data=data))#dataset needed for bootstrapping; LR_TEst_Results are the LR calculations from boot2 or 3
}
##########################################################################################
#BOOTSTRAPPING##############################################################################################################
boot_statistic_function<-function(data,Expt,y_known_quant,y_unknown_quant,AT){###For bootstrapping
 # Define the vector FirstLastSwitch
FirstLastSwitch <- character(2)
Expt <- as.numeric(Expt) 
  #This function is same as statistic_function but is used for bootstrapping. The first two commands are not used
  #data<-React(data_in,Expt)#load data corresponding to Expt2,3...
  #mydatE1<-ReactEx1(data_in)#load data from Expt1 for use to calc predicted_y value (quant. ng) based on AT RFU

## Input the expt no. into FirstLastSwitch[1]
if (Expt == 1) {
  FirstLastSwitch[1] <- "Ex_1"
} else if (Expt == 2) {
  FirstLastSwitch[1] <- "Ex_2"
} else if (Expt == 3.0) {
  FirstLastSwitch[1] <- "Ex_3.0"
} else if (Expt == 3.1) {
  FirstLastSwitch[1] <- "Ex_3.1"
} else if (Expt == 3.2) {
  FirstLastSwitch[1] <- "Ex_3.2"
} else {
  # Throw an error message if no conditions are satisfied
  stop("No Expt specified. Program failed.")
}
  
  #The Prs are recalculated as follows for expt2,3 etc:
  FirstLastSwitch[2]<-"LastH"
  LastH<-MainShiny(data$LastH,FirstLastSwitch)##this routine calculates the log normal parameters for last, first and background handlers
  FirstLastSwitch[2]<-"FirstH"
  FirstH<-MainShiny(data$FirstH,FirstLastSwitch)
  
  FirstLastSwitch<-"Back"
  Background<-MainShiny(data$Background,FirstLastSwitch)
  
  ##Calculate Prs to go into the BN
  y_SecT_Test<-LNdist(FirstH$meanlog,FirstH$sdlog,FirstH$k,y_known_quant,colour="blue")## these probabilties feed into the BN
  y_AsT_Test<-LNdist(LastH$meanlog,LastH$sdlog,LastH$k,y_known_quant,colour="blue")
  y_Bac_Test<-LNdist(Background$meanlog,Background$sdlog,Background$k,y_unknown_quant,colour="blue")
  y_AsTU_Test<-LNdist(LastH$meanlog,LastH$sdlog,LastH$k,y_unknown_quant,colour="blue")
  
  ###If y values are =1 then this resultsin infinity and the program crashes so set to 0.99
 # if (y_SecT_Test==1){y_SecT_Test<-0.99}
 # if (y_AsT_Test==1){y_AsT_Test<-0.99}
 # if (y_Bac_Test==1){y_Bac_Test<-0.99}
 # if (y_AsTU_Test==1){y_AsTU_Test<-0.99}
  

 
  
  #Now calculate the LR_test results
  #$LRPOI is the LR where there is no unknown
  #$LRTog is the LR where there are unknowns
  #Choose between BNformulaExpt2 or BNformulaExpt3 depending on the experiment
  if (Expt==2){
    LR_Test_results<-BNformulaExpt2(y_SecT_Test,y_AsT_Test,y_Bac_Test,y_AsTU_Test)
  }else{
  LR_Test_results<-BNformulaExpt3(y_SecT_Test,y_AsT_Test,y_Bac_Test,y_AsTU_Test)
  }
  return(LR_Test_results)
}

##############################
##############################
##Bootstrap 2 function does columns
###FUNCTION TO BOOTSTRAP INDIVIDUAL COLUMNS
bootstrap_and_calculate2 <- function(data, num_samples,Expt,y_known_quant,y_unknown_quant,AT) {
  num_bootstraps <- num_samples # Number of bootstrap iterations
  n <- nrow(data) # Number of rows in the data frame
  num_rows <- nrow(data) # Number of rows in the data frame
  #Initialize the results as an empty data frame with the same column names as the result data frame
  results <- data.frame(NumPOI = numeric(0), DenPOI = numeric(0), LRPOI = numeric(0), NumTog = numeric(0), DenTog = numeric(0), LRTog = numeric(0))
  bootstrapped_data <- list()  # Initialize list to store bootstrapped data 
  count=0
  #suppressMessages({
  for (i in 1:num_bootstraps) {
    # Sample indices for each column separately
    sampled_indices_LastH <- sample(1:num_rows, size = num_rows, replace = TRUE)
    sampled_indices_FirstH <- sample(1:num_rows, size = num_rows, replace = TRUE)
    sampled_indices_Background <- sample(1:num_rows, size = num_rows, replace = TRUE)
    
    # Extract values for each column based on sampled indices
    column_LastH <- data$LastH[sampled_indices_LastH]
    column_FirstH <- data$FirstH[sampled_indices_FirstH]
    column_Background <- data$Background[sampled_indices_Background]
    
    # Store sampled data in the list
    bootstrapped_data[[paste0("LastH")]] <- column_LastH
    bootstrapped_data[[paste0("FirstH")]] <- column_FirstH
    bootstrapped_data[[paste0("Background")]] <- column_Background
    
    bootstrapped_df <- as.data.frame(bootstrapped_data)  # Convert matrix to data frame
    # Calculate the statistic on the bootstrapped data
    result_to_add <- boot_statistic_function(bootstrapped_df,Expt,y_known_quant,y_unknown_quant,AT)
#record number of failed bootstraps
    if (y_unknown_quant == 0 && is.infinite(result_to_add$LRPOI)) {
      count <- count + 1
    } else if (y_unknown_quant > 0 && is.infinite(result_to_add$LRTog)) {
      count <- count + 1
    } else {
      count <- count
    }
    
    # Append the result data frame to the results data frame using rbind
    results <- rbind(results, result_to_add)
  }
    
#})
  ###Calculate quantiles for the results
  ##refers to pos results where labelled
  #choose between LRPOIonly and LRTog by selecting whether y_unknown_quant is 0 or <0
  if(y_unknown_quant==0){
    DNA_POIOnly<-as.numeric(results$LRPOI)
    DNA_POIOnly<-replace(DNA_POIOnly,is.infinite(DNA_POIOnly),NA)
    DNA_POIOnlyr<-quantile(DNA_POIOnly,probs=c(0.025,0.05,0.25,0.5,0.75,0.95,0.975),na.rm = TRUE)
  } else {
    ##### POI + UNknown result
    DNA_Tog<-as.numeric(results$LRTog)
    DNA_Tog<-replace(DNA_Tog,is.infinite(DNA_Tog),NA)
    DNA_Togyr<-quantile(DNA_Tog,probs=c(0.025,0.05,0.25,0.5,0.75,0.95,0.975),na.rm = TRUE)
  }
  
  if (exists("DNA_Togyr")) {
    DNA_Togyr=DNA_Togyr
  } else {
    DNA_Togyr=0
  }
  
  if (exists("DNA_POIOnlyr")) {
    DNA_POIOnlyr=DNA_POIOnlyr
  } else {
    DNA_POIOnlyr=0
  }
  #count the number of failed bootstraps identified by NA
  if(y_unknown_quant==0){
  count<-sum(is.na(results$LRPOI))
  }else{
  count<-sum(is.na(results$LRTog)) 
  }
  Results<-(list(DNA_POIOnlyr=DNA_POIOnlyr,DNA_Togyr=DNA_Togyr,Failed=count))
  return(Results)
}
###############################################################################################

##New Main program that doe not plot output Prs
#########################
#########################
MainShiny<-function(data,FirstLastSwitch){
data=na.omit(data)
data=as.numeric(data)
Results<-CalclnCnew(data,FirstLastSwitch)
return(list(meanlog=Results$meanlog,sdlog=Results$sdlog,k=Results$k))
}

#############NEW FUNCTION TO ACCEPT DATA############################################################
####LOAD CHOSEN DATA FROM EXPT
React<-function(data,expt){
# Load the required libraries
  
  packages <- c("readxl", "stringr", "dplyr","stringr","writexl","gridExtra","ggplot2","grid")  # List of packages you want to ensure are loaded
  
  for (pkg in packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      install.packages(pkg)
    }
    library(pkg, character.only = TRUE)
  }
  
# Read the Excel file into a list of data frames
# Read the "data" spreadsheet into a list of data frames
#Read the expt number as 1,2,3.0,3.1 or 3.2
dataframes_list <- data ###REDUNDANT read_excel(data, sheet = "data", col_names = TRUE)###reactivate this if filepath is replaced by filename##SWITCH OFF IF FILEPATH

# Assuming "ID" is the name of the ID column
unique_ids <- unique(dataframes_list$ID)

# Assuming "ID" is the name of the ID column
unique_ids <- unique(dataframes_list$ID)

# Loop through the unique IDs and create separate data frames
for (ID in unique_ids) {
  assign(ID, dataframes_list[dataframes_list$ID == ID, ])
}

# Now you can access the data frames like Ex_1, Ex_2, Ex3_0, Ex3_1, Ex3_2, etc.
# Check if Ex_1 exists and assign it to mydatE1 if true
if (exists("Ex_1")) {
  mydatE1 <- Ex_1
}

# Check if Ex_2 exists and assign it to mydatE2 if true
if (exists("Ex_2")) {
  mydatE2 <- Ex_2
}

# Check if Ex_3.0 exists and assign it to mydat3_0 if true
if (exists("Ex_3.0")) {
  mydat3_0 <- Ex_3.0
}

# Check if Ex_3.1 exists and assign it to mydat3_1 if true
if (exists("Ex_3.1")) {
  mydat3_1 <- Ex_3.1
}

# Check if Ex_3.2 exists and assign it to mydat3_2 if true
if (exists("Ex_3.2")) {
  mydat3_2 <- Ex_3.2
}

if (expt==1){
  data_out<-Ex_1
}
else if (expt==2){
  data_out<-Ex_2
}
else if (trimws(expt) == "3.0") {##need this trimws expression to prevent problem recognising 3.0
  data_out <- Ex_3.0
}

else if (expt==3.1){
  data_out<-Ex_3.1 
}
else if (expt==3.2){
  data_out<-Ex_3.2
}
return(data_out)
}
###########################################LOAD EX_1 DATA
###########################################
ReactEx1<-function(data){
#Same as REact() except we extract Ex_1 for use in regression and estiamte of predicted_y
# Read the Excel file into a list of data frames
# Read the "data" spreadsheet into a list of data frames
dataframes_list <- data#, sheet = "data", col_names = TRUE)###reactivate this if filepath is replaced by filename##SWITCH OFF IF FILEPATH



# Assuming "ID" is the name of the ID column
unique_ids <- unique(dataframes_list$ID)

# Assuming "ID" is the name of the ID column
unique_ids <- unique(dataframes_list$ID)

# Loop through the unique IDs and create separate data frames
for (ID in unique_ids) {
  assign(ID, dataframes_list[dataframes_list$ID == ID, ])
}
  data_out<-Ex_1
return(data_out)
}

##########################
############################
############################
##FUNCTIONS BELOW for all calcs
##FUNCTIONS BELOW
############################


###EXTRACT POI ROW AND SUBSET ACCORDING TO Y VALUES for LR TABLE
POI_subset<-function(vx,y,x_values){
# Initialize an empty data frame to store the results
POI_subset <- data.frame(x = numeric(0), y = numeric(0))
# Initialize an empty data frame to store the vx results
df_POI <- data.frame(x = numeric(0),y=numeric(0))
##Make a dataframe of x results

df_POI <- data.frame(x = y,y=vx)
#Extract values of vx that correspond to values of x
#input y_values for table plots here# y-values are quant of poi
#Loop through different values of  x and calculate LRs for subset
for (x_val in x_values) {
  r2<-subset(df_POI,x==x_val)   
# Append the results to the data frame
    POI_subset <- rbind(POI_subset,r2)
}
return(POI_subset)
}


#This function is used to output both the excel file and the .png file to output
#Used for expt3 only
#This function is used to output both the excel file and the .png file to output
#Used for expt3 only
exp3_plot<-function(FirstH,LastH,Background,UnknownVector,Backthresh,Lastthresh, Firstthresh,y,stringtopass,Expt,excelfilename){
  x_U_value<-0.01 ## where this is the Quant value threshold for the unknown person with POI only - pass to function POI_subset 
  Bac_POI_only<-POI_subset(Background$outC1,y,x_U_value)##To extract subset of vx values coresponding to x_values
  AsTU_POI_only<-POI_subset(LastH$outC1,y,x_U_value)
  ####To Do POI ONLY####
  if (Expt==2) {
    LRresults<-mapply(BNformulaExpt2,FirstH$outC1,LastH$outC1,Bac_POI_only$y,AsTU_POI_only$y)
  }else if (Expt==3) {
    LRresults<-mapply(BNformulaExpt3,FirstH$outC1,LastH$outC1,Bac_POI_only$y,AsTU_POI_only$y)
  }
  ## Calc LR results expt3 setting back and ASTU to 0.01 stored in $y from subset command
  LRresults_df <- data.frame(LRresults)
  # Transpose the matrix
  LRresults_df <- as.data.frame(t(LRresults))
  #LRresults_df$LRPOI is the LR output for the LRPOI with no background or unknown DNA present
  #Need to calculate BN where we dont use a continuous model
  # Use the threshold values only throughout and this calculates LRs for POI and Tog (with unknowns)
  if (Expt==2) {
    BNTresults<-BNformulaExpt3(Firstthresh,Lastthresh,Backthresh,Lastthresh)   #Calc the threshold results into BNT
  }else if (Expt==3) {
    BNTresults<-BNformulaExpt3(Firstthresh,Lastthresh,Backthresh,Lastthresh) 
  }
  
  vx=as.numeric(LRresults_df$LRPOI)  # vx contains all of the LR calculations specified by y values and unknown quant=0.01 (after adjustement from 0)
  
  x_values <- c(0.01, seq(0.5, 10, by = 0.5))#input x_values for table plots here these are QUANT KNOWN
  POI_sub<-POI_subset(vx,y,x_values)##To extract subset of vx values coresponding to x_values
  plot1=PlotresultsPOI(y,vx,BNTresults$LRPOI,Expt)#POI only plot
  ##BN results for POI+U
  plot2=PlotresultsPOIandU(y,Background,LastH,FirstH,Backthresh,Lastthresh,BNTresults$LRTog,Expt,excelfilename,POI_sub)#POI and Unknown plot
  arrangeplot<-png(stringtopass, width = 800, height = 600)  # Specify the file name and dimensions
  plotEx3<-grid.arrange(plot1, plot2,ncol = 2)
  grid.draw(arrangeplot)
  dev.off() 
  #ggsave(stringtopass, plot = plotEx3)
  return(plotEx3)
}

###Threshold function to calculate single values of quant threshold equiv. to RFU
ThreshCalc<-function(Background,LastH,FirstH,predicted_y){
BackthreshPr<-LNdist(Background$meanlog,Background$sdlog,Background$k,predicted_y,colour="blue")
LastthreshPr<-LNdist(LastH$meanlog,LastH$sdlog,LastH$k,predicted_y,colour="blue")
FirstthreshPr<-LNdist(FirstH$meanlog,FirstH$sdlog,LastH$k,predicted_y,colour="blue")
return(list(BackthreshPr,LastthreshPr,FirstthreshPr))
}
################
###################NOTE Expt is used to distinguish between exp2 and exp3 - important to use correct BN here
PlotresultsPOIandU=function(y,Background,LastH,FirstH,BackthreshPr,LastthreshPr,BNTresults,Expt,excelfilename,POI_sub){
if (Expt==2){
x_values <- c(0.01, seq(0.5, 10, by = 0.5))
}else{
x_values <- c(0.01, seq(0.5, 5, by = 0.5))
}

#input x_values for table plots here THESE ARE KNOWN QUANT VALUEs
###y=seq(0.01,10, by=0.001)
##Now calculate BNs for when POI and U are present.
##Here do multiple plots for a given quantity of unknown contributor where BackSim is background Pr and USim is last unknown handler Pr

###plot LRs for POI and Unknowns
# Create an empty plot to initialize
#windows()
#plot.new()
zx <- ggplot() +
  labs(x = "DNA POI quantity (ng)", y = "Likelihood ratio (LR)") +
  ggtitle("POI and Unknown LRs for different quant values")+
  theme(plot.title = element_text(hjust = 0.5))  # Center the plot title

# Set the y-axis to log scale
  zx <- zx + scale_y_log10()



# Initialize a data frame to store all data
all_data_i <- data.frame(x = numeric(0), y = numeric(0), z = numeric(0))#for subset calcs making tables
all_data <- data.frame(x = numeric(0), y = numeric(0), z = factor(character(0)))# for graphic calcs
### DIFFERENT FOR EXPT2 and EXPT3 switch needed
# Specify the values of z you want to loop through where z is quant of UNKNOWN DNA
#For Exp 2:::
if(Expt==2){
z_values <- c(.05, .1, .2, .3, .5, 1, 2) ##z values are here - these are the UNKNOWN contributor quants
# Loop through different values of z and add lines to the plot
for (z in z_values) {
 BackSim <- mapply(LNdist, Background$meanlog, Background$sdlog, Background$k, z)#calc Pr background for observed quant z of U
  USim <- mapply(LNdist, LastH$meanlog, LastH$sdlog, LastH$k, z)                  #calc Pr U for observed quant z of U
  BNUresults <- BNformulaExpt2(FirstH$outC1, LastH$outC1, BackSim, USim)  #Calc LRs here
  vx <- as.numeric(BNUresults$LRTog)
  
  # Create a data frame for the current z

##Use this dataframe for the subseting in tables
 dfi <- data.frame(x = y, y = vx, z = z)
  all_data_i <- rbind(all_data_i, dfi)

TableMaker(x_values,z_values,all_data_i,excelfilename,POI_sub) # make tables

  df <- data.frame(x = y, y = vx, z = factor(z))
  all_data <- rbind(all_data, df)
}
###For Exp 3:::Multiple plots are plotted here
}else if(Expt==3){
z_values <- c(0.05, 0.1, 0.15, 2, 3, 5) ##z values are here - these are the unknown contributor quants - these can be different to expt 2
# Loop through different values of z and add lines to the plot
#FirstH$outC1[is.na(FirstH$outC1) | FirstH$outC1 < 0.01] <- 0.01 ## replace small values or NA with default value of 0.01
#if (any(is.na(FirstH$outC1) | FirstH$outC1<0.01)) { FirstH$outC1=0.01}
for (z in z_values) {
  BackSim <- mapply(LNdist, Background$meanlog, Background$sdlog, Background$k, z)#calc Pr background for observed quant z of U
  USim <- mapply(LNdist, LastH$meanlog, LastH$sdlog, LastH$k, z)                  #calc Pr U for observed quant z of U
  BNUresults <- BNformulaExpt3(FirstH$outC1, LastH$outC1, BackSim, USim)  #Calc LRs here
  vx <- as.numeric(BNUresults$LRTog)
  #browser()
  # Create a data frame for the current z

##Use this dataframe for the subseting in tables
 dfi <- data.frame(x = y, y = vx, z = z)
  all_data_i <- rbind(all_data_i, dfi)

TableMaker(x_values,z_values,all_data_i,excelfilename,POI_sub)


###
  df <- data.frame(x = y, y = vx, z = factor(z))
  all_data <- rbind(all_data, df)
}
} else {
all_data <-0#if there is no expt indicator the program fails here
}

# Add lines to the plot with a legend
zx <- zx + geom_line(data = all_data, aes(x = x, y = y, color = z)) +
  scale_color_manual(values = rainbow(length(z_values)), 
                     labels = z_values,
                     name = "Quants for U")+
 if (Expt == 3) {
    scale_x_continuous(limits = c(0, 2))  # Set x-axis limits from 0 to 5 for Expt=3
  } else {
    scale_x_continuous(limits = c(0, 10))  # Set x-axis limits from 0 to 10 for other Expt values
  }


if (BNTresults >.1){
BNTresults <- round(BNTresults, 1) #(round to one sig figure - these are the non cont. calcs using threshold data)
}
zx<-zx+
geom_text(
   aes(x = 2, y = 0.005, label = str_wrap(paste("LR POI and Unknown using qualitative model =", BNTresults), width=20)),
    vjust = 1.5,  # Position the text halfway vertically
    hjust = 0,  # Position the text to the left
    lineheight = 1  # Adjust the line spacing for line wrapping
  )



return(zx)
}

TableMaker<-function(x_values,z_values,all_data_i,excelfilename,POI_sub){
#############Notes - this works#########################################################

# Initialize an empty data frame to store the results
results_table <- data.frame(x = numeric(0), y = numeric(0), z = numeric(0))
###x and z values are input through the function header
#x_values <- c(0.01, seq(0.5, 10, by = 0.5))
#z_values <- c(0.04, 0.1, 0.2, 0.3, 0.5, 1, 2)  # Add more values as needed

 #Loop through different values of z and x and calculate BNUresults
for (z_val in z_values) {
for (x_val in x_values) {
  r2<-subset(all_data_i,x==x_val & z==z_val)   
    # Append the results to the data frame
    results_table <- rbind(results_table,r2)
  }
}
#append the POI_sub values
df_sub <- cbind(POI_sub, z=0)##need to add z=0 to dataframe
results_table <- rbind(df_sub,results_table)

#browser()
# Load the tidyr package
library(tidyr)

# Pivot the results_table data frame
reshaped_data <- pivot_wider(results_table, names_from = x, values_from = y)

# Print the reshaped data
#print(reshaped_data)
suffix <- ".xlsx"
# Define the file path where you want to save the Excel file
excel_file_path <- paste0(excelfilename, suffix)  # Change this to your desired file name

# Save the reshaped data as an Excel file
write_xlsx(reshaped_data, excel_file_path)

# Print a message to confirm the file has been saved
cat("Data saved as an Excel file:", excel_file_path, "\n")

return()
}

##Make a plot function for LR values### POI ONLY
PlotresultsPOI=function(y,vx,BNTresults,Expt){
df <- data.frame(y = y, vx = vx)
# Create a line plot
#windows()
#plot.new()
zx=ggplot(df, aes(x = y, y = vx)) +
  geom_line() +
  labs(x = "DNA total quantity (ng)", y = "Likelihood ratio (LR)") +  # Customize axis labels
  ggtitle("POI only recovered") + # Add a plot title
theme(plot.title = element_text(hjust = 0.5))  # Center the plot title
# Set the y-axis to log scale
  zx <- zx + scale_y_log10()+ if (Expt == 3) {
    scale_x_continuous(limits = c(0, 2))  # Set x-axis limits from 0 to 5 for Expt=3
  } else {
    scale_x_continuous(limits = c(0, 10))  # Set x-axis limits from 0 to 10 for other Expt values
  }



##########################################

BNTresults <- round(BNTresults, 1) #(round to one sig figure)

zx<-zx+
geom_text(
    aes(x = min(y), y = log10(median(vx)), label = str_wrap(paste("LR POI only using qualitative model =", BNTresults), width=20)),
    vjust = 0.5,  # Position the text halfway vertically
    hjust = 0,  # Position the text to the left
    lineheight = 1  # Adjust the line spacing for line wrapping
  )
return(zx)
}

#########################
#########################
Main<-function(data,y,colour,Expt,FirstLastSwitch){
data=na.omit(data)
data=as.numeric(data)
#set the interval to calculate for plots
#y=seq(0,30, by=0.1)
Results<-CalclnCnew(data,FirstLastSwitch)
plotLiLN(Results$meanlog,Results$sdlog,Results$k,y,colour,Expt)
#To retrieve probabilities
outC1=mapply(LNdist,Results$meanlog,Results$sdlog,Results$k,y)

return(list(meanlog=Results$meanlog,sdlog=Results$sdlog,k=Results$k,outC1=outC1))
}
#################################################
###################################################

CalcUthresh <- function(mydat, x_value, new_file_path) {
x_value=2*x_value # 2*AT RFU equivalent
#remove rows wih zeros otherwise regression will fail
mydat <- mydat[!(mydat$TotalQ == 0 | mydat$AdjAvgRFU == 0), ]


#remove rows wih zeros otherwise regression will fail
#mydat <- mydat[rowSums(mydat[c("TotalQ", "AdjAvgRFU")]) != 0, ]
#if (any(mydat$TotalQ == 0)) {
#  mydat$TotalQ[mydat$TotalQ == 0] <- 0.01  # Set zero values in TotalQ to 0.01
#}

#if (any(mydat$AdjAvgRFU == 0)) {
#  mydat$AdjAvgRFU[mydat$AdjAvgRFU == 0] <- x_value  # Set zero values in AdjAvgRFU to x_value
#}
####
  # Fit the linear regression model
  lm_model <- lm(log10(TotalQ) ~ log10(AdjAvgRFU), data = mydat)
if (!is.null(new_file_path)) {###TEST FOR SHINY WHERE WE DO NOT MAKE GRAPH
 
  # Create a scatter plot of regression data
  p <- ggplot(mydat, aes(x = log10(AdjAvgRFU), y = log10(TotalQ))) +
    geom_point(color = "blue") +
    labs(x = "log10 Average (adj) RFU", y = "log10 total DNA quantity (ng)") +
    geom_smooth(method = "lm", se = FALSE, color = "red")

  # Get the coefficients and adjusted R-squared value from the model
  coef_intercept <- coef(lm_model)[1]
  coef_slope <- coef(lm_model)[2]
  adjusted_r_squared <- summary(lm_model)$adj.r.squared

  # Create a label with the coefficients and adjusted R-squared value on different lines
  label_text <- paste("Intercept coef: ", round(coef_intercept, 2), "\n",
                      "Slope coef: ", round(coef_slope, 2), "\n",
                      "Adj R-squared: ", round(adjusted_r_squared, 2))

  # Add the text annotation using grid.text
  p <- p + annotation_custom(
    grob = textGrob(label_text, x = 0.6, y = 0.25, just = c("left", "bottom"), gp = gpar(fontsize = 12))
  )

  # Remove the existing file, if it exists
file_name <- "reg_plot_Ex1.png"
full_file_path <- file.path(new_file_path, file_name)
if (file.exists(full_file_path)) {
  file.remove(full_file_path)
}

# Save the plot as a PNG file
ggsave(filename = full_file_path, plot = p, width = 8, height = 6, units = "in", dpi = 300)


  # Calculate predicted_y (omitted for brevity)
#}
#x_value=100 #the ave RFU
} ####END IF IS NOT NULL CONDITION

predicted_y <- predict(lm_model, newdata = data.frame(AdjAvgRFU = x_value))
##where predicted_y = 10^predicted_y. The lower threshold quant value
predicted_y = 10^predicted_y
}
#############################################################

CalclnCnew<-function(data,FirstLastSwitch){ #for lognormal
####This routine is for the full beta binomial method currently switched off
#Three beta distribution look up tables for LastH, FirstH and Back respectively
##LastH
function(lookupTables){
# Create a matrix for the lookup table
LastH_lookup_table <- matrix(
  c(
    9.373074, 4.579944, 2.449359, 1.436104, 1.67986,   # Alpha values
    0.2941873, 0.1939859, 0.1395279, 0.1085753, 0.1213885  # Beta values
  ),
  nrow = 2,   # 2 rows for alpha and beta
  byrow = TRUE,  # Fill matrix by row
  dimnames = list(
    c("alpha", "beta"),  # Row names
    c("Ex_1", "Ex_2", "Ex_3.0", "Ex_3.1", "Ex_3.2")  # Column names
  )
)



# Create a matrix for the lookup table
FirstH_lookup_table <- matrix(
  c(
    NA, 5.404182, 1.957904, 1.502473, 1.348317,   # Alpha values for LastH
    NA, 0.7112159, 0.8681403, 1.503382, 1.442789  # Beta values for LastH
  ),
  nrow = 2,   # 2 rows for alpha and beta
  byrow = TRUE,  # Fill matrix by row
  dimnames = list(
    c("alpha", "beta"),  # Row names
    c("Ex_1", "Ex_2", "Ex_3.0", "Ex_3.1", "Ex_3.2")  # Column names
  )
)

# Create a matrix for the lookup table
Back_lookup_table <- matrix(
  c(
    1.490608,	1.221976,	0.9843282,	0.8277924,	0.7911678, # Alpha values for LastH
    0.8621806,	1.263275,	1.859354,	1.157104,	1.256673
  # Beta values for LastH
  ),
  nrow = 2,   # 2 rows for alpha and beta
  byrow = TRUE,  # Fill matrix by row
  dimnames = list(
    c("alpha", "beta"),  # Row names
    c("Ex_1", "Ex_2", "Ex_3.0", "Ex_3.1", "Ex_3.2")  # Column names
  )
)

if (FirstLastSwitch[2] == "LastH") {
  alpha <- LastH_lookup_table["alpha", FirstLastSwitch[1]]
  beta <- LastH_lookup_table["beta", FirstLastSwitch[1]]
} else if (FirstLastSwitch[2] == "FirstH") {
  alpha <- FirstH_lookup_table["alpha", FirstLastSwitch[1]]
  beta <- FirstH_lookup_table["beta", FirstLastSwitch[1]]
} else if (FirstLastSwitch[2] == "Back") {
  alpha <- Back_lookup_table["alpha", FirstLastSwitch[1]]
  beta <- Back_lookup_table["beta", FirstLastSwitch[1]]
} else {
  stop("Failure in look up table")
}
}
######Function switched off
### Use Jefrreys or Laplace binomial method ie c+1/n+2
alpha=1
beta=1
 # FirstLastSwitch specified firstH or LastH, along with Ex_
data=as.numeric(unlist(data))
##calculate k  which is the proportion of zeros in the dataset.
##k=sum(data>0.0)#count no of data>0 # only positive data are included
#for expt2 some data are low quant hence switch to 0.1
#set data threshold:
Datatrhesh=0.001
ksum=sum(data>Datatrhesh)#count no of data>0 # only positive data are included
##Specify priors a and b from look up tables

k=1-((ksum+alpha)/(length(data)+alpha+beta))#so this is the binomial adj Pr

#set default fp - from lab3_NGM background expt_1
meanlog_default<--1.927072
meansd_default<-0.995147
fx_default<-c(meanlog_default,meansd_default) #set fp

ndat<-subset(data,data>Datatrhesh) #extract data>0.001

# Delete the error log file if it exists
#if (file.exists("error_log.txt")) {
#  file.remove("error_log.txt")
#}

# Close the sink to redirect the output back to the console
#sink(type = "message")

# Open the file for writing errors (will overwrite if it exists)
#sink("error_log.txt")

##now check for obvious causes of failure of fitdist()
if (!exists("ndat") || !is.numeric(ndat) || length(ndat) <= 1) {
  fx <- fx_default # keep default value and end the loop
} else {
  # Test whether fp is viable
  fpres <- try({
    fp <- fitdist(ndat, "lnorm")
    as.numeric(unlist(fp$estimate))
  }, silent = TRUE)
  
  if (inherits(fpres, "try-error")) {
    fx <- fx_default # If an error occurred, assign default value
  } else {
    fx <- fpres # Use the calculated value
  }
}

# Close the sink to redirect the output back to the console
#sink(type="message")


# Delete the error log file if it exists
#if (file.exists("error_log.txt")) {
#  file.remove("error_log.txt")
#}


#adopt surviving value of fx
#diagnostics below if needed - switched off
#windows()
#plot.new()
#par(mfrow = c(2, 2))
#denscomp(fp)
#cdfcomp(fp)
#qqcomp(fp)
#ppcomp(fp)
#print(summary(fp))
Results<-(list(k=k,data=data,meanlog=fx[1],sdlog=fx[2]))
return(Results)
}
#####plot graph for lognormal curves###################
########################
plotLiLN <- function(meanlog, sdlog, k, y, colour,Expt) {
  out <- mapply(LNdist, meanlog, sdlog, k, y)
  
  # Use switch for different plot styles based on the 'colour' argument
  if (colour == "green") {
    plot(y, out, type = "l", col = "green", lwd = 2, ylim = c(0, 1),log = "x",
         xlab = "DNA recovered (ng)", ylab = expression("Probability of transfer"))
# Add a legend

if (Expt==3) {
    legend("topright", legend = c("Last handler", "Secondary", "Background"),
           col = c("green", "blue", "orange"), lty = c(1, 5, 1), lwd = 2)
} else {


legend("topright", legend = c("Last handler", "First handler", "Background"),
           col = c("green", "blue", "orange"), lty = c(1, 5, 1), lwd = 2)

}

  } else {
    # Adjust line type based on 'colour'
    lineT <- ifelse(colour == "blue", 5, 1)
    lines(y, out, col = colour, lty = lineT, lwd = 2)
  }
  
  # Add a legend here if needed
}


###################################
##the log normal formula###
LNdist<-function(meanl,sdl,k,y,colour){
LN<-plnorm(y,meanlog=meanl,sdlog=sdl)
expR<-1-(k+((1-k)*LN))
if (expR<0.00001) {expR=NA}##default lower PR
return(expR)
}

##BN FORMULAE: Function LR formulae substitute for BN
#SecT=First Handler POI; AsT=Second (last) handler POI;ASTU=Pr transfer of unknown person which is same distribution as AsT and Bac=Pr background
##Note that BackthreshPr is the value of the Pr of finding an unknown person as background >threshold value, when none is observed in the "POI only" formula
##LastthreshPr is Pr of finding a last unknown handler when none is observed under Hd
##Expt2 BN
BNformulaExpt2<-function(SecT,AsT,Bac,AsTU){
  #To prevent failure of calculation where SecT=0, here we assign a very small value to SecT=0.01 - this is a conservative under Hd, because it increases Pr
  #if (any(SecT == 0)) {SecT=0.01}
  
  ###If y values are =1 then this resultsin infinity and the program crashes so set to 0.99
  #if (any(SecT==1)){SecT<-0.99}
 # if (any(AsT==1)){AsT<-0.99}
 # if (any(Bac==1)){Bac<-0.99}
 # if (any(AsTU==1)){AsTU<-0.99}
  
U<-(AsTU*Bac)+(AsTU*(1-Bac))+(Bac*(1-AsTU))# given an unknown is observed - this is used only where POI + U are in results
NUM<-(SecT*(1-AsT))+AsT
#POI ONLY -no U or background observed. Note that threshold values are used for 1-background and 1-last handler Prs here
NumPOI<-NUM*(1-Bac)
DenPOI<-SecT*(1-Bac)*(1-AsTU)
#Calculate NumU and DenU for unknown DNA only
NumU<-(1-AsT)*Bac*(1-SecT)
DenU<-((AsTU)+(Bac*(1-AsTU)))*(1-SecT)
#NoDNA_Num amd Den - no DNA observed
NoDNA_Num<-(1-AsT)*(1-Bac)*(1-SecT)
NoDNA_Den<-(1-AsTU)*(1-Bac)*(1-SecT)
#POI and Unknown DNA present ie we calculate Pr bacground and last handler unknown
NumTog<-NUM*Bac
DenTog<-SecT*U
LRPOI<-NumPOI/DenPOI
#if (any(is.infinite(LRPOI))) {browser()}#debugging
LRTog<-NumTog/DenTog
#if (any(is.infinite(LRTog))) {browser()}#debugging
#check sums of probabilities=1
#Num
PrSumHp<-NumPOI+NumTog+NumU+NoDNA_Num
#Den
PrSumHd<-DenPOI+DenTog+DenU+NoDNA_Den
ResLR<-(list(NumPOI=NumPOI, DenPOI=DenPOI, LRPOI=LRPOI, NumTog=NumTog, DenTog=DenTog,LRTog=LRTog,NumU=NumU,DenU=DenU,NoDNA_Num=NoDNA_Num,NoDNA_Den=NoDNA_Den,PrSumHp=PrSumHp,PrSumHd=PrSumHd,
             SecT=SecT,AsT=AsT,Bac=Bac,AsTU=AsTU))
return(ResLR)
}

##Expt3 BN

BNformulaExpt3<-function(SecT,AsT,Bac,AsTU){
  #To prevent failure of calculation where SecT=0, here we assign a very small value to SecT=0.01 - this is a conservative under Hd, because it increases Pr
 # if (any(SecT == 0)) {SecT=0.01}
  
  ###If y values are =1 then this resultsin infinity and the program crashes so set to 0.99
 # if (any(SecT==1)){SecT<-0.99}
 # if (any(AsT==1)){AsT<-0.99}
 # if (any(Bac==1)){Bac<-0.99}
 # if (any(AsTU==1)){AsTU<-0.99}
 # 
  
  
  U<-(AsTU*Bac)+(AsTU*(1-Bac))+(Bac*(1-AsTU))# given an unknown is observed - this is used only where POI + U are in results
#POI ONLY -no U or background observed. Note that threshold values are used for 1-background and 1-last handler Prs here
NumPOI<-AsT*(1-Bac)
DenPOI<-SecT*(1-Bac)*(1-AsTU)
LRPOI<-NumPOI/DenPOI
#if (is.infinite(LRPOI)) {browser()}#debugging
#Calculate NumU and DenU for unknown DNA only
NumU<-(1-AsT)*Bac
DenU<-((AsTU)+(Bac*(1-AsTU)))*(1-SecT)
#NoDNA_Num amd Den - no DNA observed
NoDNA_Num<-(1-AsT)*(1-Bac)
NoDNA_Den<-(1-AsTU)*(1-Bac)*(1-SecT)
#POI and Unknown DNA present ie we calculate Pr bacground and last handler unknown
NumTog<-AsT*Bac
DenTog<-SecT*U
LRTog<-NumTog/DenTog
#if (is.infinite(LRTog)) {browser()}#debugging
#check sums of probabilities=1
#Num
PrSumHp<-NumPOI+NumTog+NumU+NoDNA_Num
#Den
PrSumHd<-DenPOI+DenTog+DenU+NoDNA_Den
ResLR<-(list(NumPOI=NumPOI, DenPOI=DenPOI, LRPOI=LRPOI, NumTog=NumTog, DenTog=DenTog,LRTog=LRTog,NumU=NumU,DenU=DenU,NoDNA_Num=NoDNA_Num,NoDNA_Den=NoDNA_Den,PrSumHp=PrSumHp,PrSumHd=PrSumHd,
              SecT=SecT,AsT=AsT,Bac=Bac,AsTU=AsTU))
return(ResLR)
}


##end functions

