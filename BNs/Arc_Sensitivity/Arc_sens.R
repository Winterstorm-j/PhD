library("dplyr")
library("bnlearn")
library("gRain")
# library("gRaven")


originalNetwork  <- read.net("BNs/Arc_Sensitivity/simpleBNexample.net")

#bn_grain <- loadHuginNet("BNs/Arc_Sensitivity/simpleBNexample.net")

origPlot <- graphviz.plot(originalNetwork,
  groups = list(
    c("HpHd", "Activity2", "Activity3"),
    c("TPPR4", "TPPR5", "TPPR6", "Results7")
  )
)

graphviz.chart(originalNetwork,
                            type = "barprob",
                            grid = TRUE, 
                            bar.col = "darkgreen",
                            strip.bg = "lightskyblue")

# arcArray = list()
# arcs <- arcs(originalNetwork)  
# arcBool <- if_else(in.degree(originalNetwork, arcs[,"to"]) >= 1, TRUE, FALSE)
# arcs <- cbind(arcs,arcBool)

# iterArc <- function(BN) {
#     arcs <- arcs(BN)  
#     for(arc in arcs) {
#         
#         }
#     }
set.seed(123)  # For reproducibility
# }
arcs <- arcs(originalNetwork)
nrow <- length(arcs[, 1])
origBN <- originalNetwork
sim_data <- rbn(origBN, n = 100000)

for (i in 1:nrow) {
  print(i)
  parent <- arcs[[i, "from"]]
  child <- arcs[[i, "to"]]
  set.seed(123)  # For reproducibility
  bn_structure <- bn.net(origBN)
  newBN <- drop.arc(bn_structure, parent, child)
  graphviz.plot(newBN)
  new_data <- cptable(child,
    levels=levels(sim_data[, which(names(sim_data) == child)]),
    values = table(sim_data[ , which(names(sim_data) == child)])/nrow(sim_data)
  )

  replace_cpt(as.grain(origBN), new_data)
  tempBN_fitted <- bn.fit(newBN, sim_data) #[ , - which(names(sim_data) == child)])

  tempBN_pred  <- predict(tempBN_fitted,
    data = data.frame(Results7 = factor("Yes", levels = c("Yes", "No"))),
    node = "HpHd",
    method = "bayes-lw",
    n = 10000,
    prob = TRUE
  ) # Generate 10,000 samples if results true

  # Compute empirical marginal distributions
  marginal_HpHd <- attributes(tempBN_pred)$prob

  cat("Empirical Marginals via Monte Carlo Sampling:\n")
  cat("P(HpHd):\n")
  print(marginal_HpHd)

  print(marginal_HpHd[1]/ marginal_HpHd[2])

}


#bnlearn::BF(numerator, denom) computes bayes factor between two networks



# P(C∣Remaining Parents)= ∑ P(C∣P,Remaining Parents)⋅P(P)

new_prob <- querygrain(tempBN, nodes = child, type = "conditional")* querygrain(tempBN, nodes = parent, type = "conditional")

# Fit the model with the new structure
model <- bn.fit(net, data)

# View the updated CPDs
model$A