library("dplyr")
library("bnlearn")
# library("gRain")
# library("gRaven")

originalNetwork  <- read.net("BNs/Arc_Sensitivity/simpleBNexample.net")

# bn_grain <- loadHuginNet("BNs/Arc_Sensitivity/simpleBNexample.net")

origPlot <- graphviz.plot(originalNetwork,
  groups = list(
    c("HpHd", "Activity2", "Activity3"),
    c("TPPR4", "TPPR5", "TPPR6", "Results7")
  )
)

origChart <- graphviz.chart(originalNetwork,
                            type = "barprob",
                            grid = TRUE, 
                            bar.col = "darkgreen",
                            strip.bg = "lightskyblue")

arcArray = list()
arcs <- arcs(originalNetwork)  
arcBool <- if_else(in.degree(originalNetwork, arcs[,"to"]) >= 1, TRUE, FALSE)
arcs <- cbind(arcs,arcBool)

iterArc <- function(BN) {
    nodes <- nodes(BN)
    for(node in nodes) {
        parentNodes <- bnlearn::parents(BN, node)
        inDeg <- bnlearn::in.degree(BN, node)
        if(inDeg >= 1) {
            arcArray[node] <- TRUE }
        else {
            arcArray[node] <- FALSE
        }
    }

}

iterArc(originalNetwork)
arcArray
