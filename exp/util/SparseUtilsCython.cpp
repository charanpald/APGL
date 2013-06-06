

void partialReconstructValsPQCpp(int* rowInds, int*colInds, double* P, double* Q, double* values, int size, int numCols) { 
    /*
    * Go through and reconstruct PQ^T for the given indices 

    */
    int i, j; 
    int p, q; 
    double sum; 

    for(i=0;i<size;i++) {
        sum = 0; 
        p = rowInds[i]*numCols; 
        q = colInds[i]*numCols;
        for(j=0;j<numCols;j++) { 
            //P[rowInds[i], :]^T Q[colInds[i], :]
            sum += P[p + j]*Q[q + j];
            }
 
        values[i] = sum;         
        }
    }