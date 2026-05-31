>**This is how PCA workflow**
#pca #unsupervised #ml 

**Step-by-step process** Jonathon Shlens, 2014 [2]:

1. **Data Organization:** Arrange data as an m × n matrix, where m is the number of measurement types and n is the number of samples
    
2. **Data Preprocessing:** Subtract off the mean for each measurement type (centering the data)
    
3. **Covariance Analysis:** Calculate the covariance matrix, where diagonal terms represent variance of particular measurements and off-diagonal terms represent covariance between measurement types
    
4. **Mathematical Solution:** Calculate either:
    
    - The eigenvectors of the covariance matrix, OR
    - The Singular Value Decomposition (SVD) of the data matrix
5. **Component Extraction:** The principal components are the eigenvectors, rank-ordered by their associated eigenvalues (which represent variance)

**Key Insight:** Jonathon Shlens, 2014 explains that PCA assumes “directions with largest variances in our measurement space contain the dynamics of interest,” making it effective for finding the most meaningful basis to re-express data while filtering out noise.

The method is **adaptive** Ian T Jolliffe & Jorge Cadima, 2016 because the new variables are “defined by the dataset at hand, not a priori,” and **incremental** because the best q+1 dimensional solution builds upon the best q-dimensional solution.