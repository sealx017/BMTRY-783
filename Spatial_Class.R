suppressWarnings(devtools::install_github('sealx017/SpaceANOVA', force = T))
require(SpaceANOVA)

require(spatstat)
require(tidyr)
require(dplyr)
require(fda.usc)
require(ggplot2)
require(cowplot)
require(gridExtra)

#----------------------------------------
## Loading and visualizing the data
#----------------------------------------

# loading the data from SpaceANOVA package
data("IMC_T1DM") 

# checking the data structure
knitr::kable(head(IMC_T1DM), format="markdown") 

# checking cell count in each image per subject per group
Data_summary = IMC_T1DM %>% group_by(Group, ID, imageID) %>% 
  summarize(cell_count = n())
Data_summary

# checking number of images per subject ID
Data_summary[, 1:3] %>% summarize(Image_count = n())

# checking overall cell distribution
table(IMC_T1DM$cellType)

# visualizing cell distribution in 4 images from subject: 6126, 
# using SpaceANOVA
palette = c("darkorchid1","red", "cyan", "grey", "blue", "green") #assign colors to cell types 
Plot.cellTypes(data = IMC_T1DM, ID = "6126", palette = palette)

#----------------------------------------
## Testing if cell type counts vary between groups: Non-diabetic and Onset
#----------------------------------------

# creating a data frame with count of each cell type in each image
Celltype_data = IMC_T1DM %>% group_by(Group, ID, imageID, cellType) %>% 
  summarize(cell_count = n())
head(Celltype_data)

# converting the data frame from long to wide data format
Celltype_data_long = Celltype_data %>% pivot_wider(names_from = cellType, 
                     values_from = cell_count)
head(Celltype_data_long)
Celltype_data_long$total = rowSums(Celltype_data_long[, -c(1:3)], na.rm = T) # add the total cell count as a column
head(Celltype_data_long) # in some images a cell type might be absent, givinng us 0 or NA count

# for each cell type fit Negative Binomial GLM to test association with groups
m = list() # storing the fitted models in this list object
i = 1
for(cellType in c("alpha", "beta", "delta", "Tc", "Th")){
  Celltype_specific_data = Celltype_data_long[, c("Group", "ID", "imageID",
                                                  cellType, "total")]
  colnames(Celltype_specific_data)[4] = "cell_count" # renaming the cell type column for simplicity
  m[[i]] <- glm.nb(cell_count ~ Group + offset(log(total)), 
                data =  Celltype_specific_data) # log(total) is used as an offset
  i = i + 1
}

# presenting the results nicely
summary(m[[1]])
store_results = matrix(0, 5, 3)
for(i in 1:5){
  store_results[i, ] = c(summary(m[[i]])$coefficients[1, 1],
                         summary(m[[i]])$coefficients[2, 1],
                         summary(m[[i]])$coefficients[2, 4])
}
store_results = as.data.frame(store_results) # change from matrix to dataframe
colnames(store_results) = c("Intercept", "Onset", "p-value")
rownames(store_results) = c("alpha", "beta", "delta", "Tc", "Th")
store_results

# multiple testing correction using Benjamini-Hochberg procedure
store_results$Adj_pvalue = p.adjust(store_results$`p-value`, method = "BH")
store_results

#----------------------------------------
## Testing if co-occurrence of pairs of cell types 
## vary between groups: Non-diabetic and Onset
#----------------------------------------

# how to use spatstat package?
one_subject = IMC_T1DM[IMC_T1DM$ID == "6126", ]
one_subject_one_image = one_subject[one_subject$imageID == "E06", ]
X =  one_subject_one_image$x
Y =  one_subject_one_image$y
ppp_object = ppp(X, Y, window = owin(c(min(X),max(X)),c(min(Y),max(Y))))
marks(ppp_object) = one_subject_one_image$cellType
plot(ppp_object) # plotting the point pattern object


# slightly nicer plot but not nice enough
ppp_object_split <- split(ppp_object, reduce = TRUE)
plot(ppp_object_split[[1]], markscale=.01,
     main = "", cols = palette[1], bg = palette[1])
for(i in 2:6){
  plot(ppp_object_split[[i]], markscale=.01,
       add = TRUE, col = palette[i], bg = palette[i])
}

Plot.cellTypes(data = IMC_T1DM, ID = "6126", imageID = "E02", 
               palette = palette) # using SpaceANOVA


# estimating K function and g function for this image
fixed_r = seq(0, 100, by = 5) # specify grid of radii
estimated_K = Kcross(ppp_object, r = fixed_r, 
              correction = "isotropic") # for a single pair of cell types
estimated_K_all = alltypes(ppp_object, Kcross, r = fixed_r, 
                  correction = "isotropic") # for all pairs of cell types
estimated_K_all
estimated_g_all = pcf(estimated_K_all, spar=1, method="c", divisor ="d")
plot(estimated_g_all[1, 1], main = "(alpha, alpha)")
plot(estimated_g_all[1, 2], main = "(alpha, beta)")
plot(estimated_g_all[4, 4], main = "(Others, Others)")

# using SpaceANOVA to estimate g functions for each pair of cell types and compare between groups
Final_result = All_in_one(data = IMC_T1DM, fixed_r = seq(0, 100, by = 5), 
                          Summary_function = "g", Hard_ths = 10, 
                          homogeneous = TRUE, interaction_adjustment = TRUE, 
                          perm = TRUE, nPerm = 5, cores = 1)
p_res = p_extract(Final_result)
Univ_p = p_res[[1]]
#Mult_p = p_res[[2]]
print(Univ_p)

Plot.heatmap(Univ_p, main = "SpaceANOVA Univ.")
# (beta, beta) and (delta, delta) appear to be significant

# extracting functional profiles
Pointwise_Fvals = Final_result[[1]][[2]]
Functional_results = Final_result[[2]]

# Pair: (beta, beta)
Plot.functions(Functional_results, 
               pair = c("beta", "beta"), 
               Fvals = Pointwise_Fvals)

# Pair: (delta, delta)
Plot.functions(Functional_results, 
               pair = c("delta", "delta"), 
               Fvals = Pointwise_Fvals)
