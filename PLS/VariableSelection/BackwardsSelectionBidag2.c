/* ————————————————————————————————————————————————————————————————————————————————————————————————
Backwards elimination algorithm using Bidiag2.
———————————————————————————————————————————————————————————————————————————————————————————————————
This is a MEX function which uses the Bidiag2 PLS algorithm to estimate the cross-validated root-
mean-squared-error, RMSEcv, of an initial design matrix X, regressed upon y, together with every
possible single inactivation of a column. The function returns a matrix with RMSEcv as a function
of variable selection and PLS component where the first row corresponds to the initial variable 
selection and every following row corresponds to the inactivation of variable 1 to n.

The function takes 5 inputs:
* Input 1: a [m x n] design matrix with observations as rows and variables as columns.
* Input 2: a [m x 1] vector with response values (multiple responses is not supported).
* Input 3: a [1 x 1] scalar 'A' specifying the maximum number of PLS components to calculate.
* Input 4: a [1 x 1] scalar 'MaxIter' specifying how many MonteCarlo CV iterations should be done.
* Input 5: a [1 x 1] scalar 'NoValObs' specifying how many unique responses should be in one validation fold.

The function outputs 1 variable:
* Output 1: a [(n+1) x A] matrix with RMSEcv. Variable selection along rows and components
along columns.

Example on how to compile and run from Matlab:
% Compile .C to .mexw64
>> mex -largeArrayDims -lmwblas BackwardsSelectionBidag2.c

% Run from Matlab when compiled:
>> X = rand(10000, 256);
>> y = rand(10000, 1);
>> A = 10;
>> MaxIters = 10;
>> Valobs = round(length(unique(y))/4);

>> [ RMSEcv ] = BackwardsSelectionBidag2( X , y, A, MaxIters, Valobs );

Example of compatible C compilers:
* Microsoft Visual C++ 2013 Professional (C)
* Microsoft Visual C++ 2015 Professional (C)
* Intel Parallel Studio XE 2017

Written 2017-08-14 by
petter.stefansson@nmbu.no
———————————————————————————————————————————————————————————————————————————————————————————————— */

#include <mex.h>	// Needed to communicate with matlab
#include <blas.h>	// Needed for blas functions
#include <string.h> // Needed to avoid compiler warning due to memcpy when using old compilers
#include <math.h>   // Needed to take the square-root (sqrt)
#include <time.h>   // Needed for counting CPU clock cycle which is used to set seed for rand()

/* ——————————————————————————————————— Function declarations ——————————————————————————————————— */
void PLS(const double *X, const double *y, int A, size_t m, size_t n, size_t p, double *beta, 
	double *B, double *w, double *wn, double *W, double *rho, double *rhoi, double *d, double *tty, 
	double *Xt, double *Ww, double *WWw, double *theta, double *thetai, double *Tt);

int randr(unsigned int min, unsigned int max);

void MarkAsVal(bool *IsVal, int NoValObs, size_t m, size_t *valm, size_t *trainm, const double *y);

void ExtractXandY(const double *X, const double *y, double *Xtrain, double *Xval, double *ytrain,
	double *yval, size_t m, size_t n, bool *IsVal);

void Pred(double *Xval, double *beta, double *yval, size_t n, size_t valm, int A, double *yhat, 
	double *RMSEcv, int cvIter, int MaxIters, int ShavingIndex);

void SwapCols(double *Xtrain, double *Xval, size_t trainm, size_t valm, size_t n, int ColToShave);

/* ——————————————————————————————————— Matlab gateway start ———————————————————————————————————— */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	/* —————————————————————————— Variable type and name declaration ——————————————————————————— */
	const double *X, *y;
	double *Xtrain, *Xval, *ytrain, *yval, *beta, *yhat, *B, *w, *wn, *W, *rho, *rhoi, *d, *tty,
		   *Xt, *Ww, *WWw, *theta, *thetai, *Tt;

	int cvIter, NoValObs, MaxIters, A, ColToShave;
	bool *IsVal;
	size_t m, n, p, trainm, valm;

	/* ————————————————————  Get pointers to the input variables from Matlab ——————————————————— */
	X = mxGetPr(prhs[0]);			        // First input (X matrix).
	y = mxGetPr(prhs[1]);			        // Second input (Y vector).

	/* ————————————————————  Get input scalar values containing PLS settings ——————————————————— */
	A        = (int)mxGetScalar(prhs[2]);   // Third input (max number of components to calculate)
	MaxIters = (int)mxGetScalar(prhs[3]);   // Fourth input (Max cv-iters)
	NoValObs = (int)mxGetScalar(prhs[4]);   // Fifth input (Number of validation observations)

	/* ——————————————————————— Get the dimensions of the input variables ——————————————————————— */
	m = mxGetM(prhs[0]);                    // Number of rows in    X.
	n = mxGetN(prhs[0]);                    // Number of columns in X.
	p = mxGetN(prhs[1]);                    // Number of columns in y.

	/* ——————————————————————————— If input A is larger than n, let A = n —————————————————————— */
	if ((A > n) || (A < 1)) { A = n; }

	/* ——————————————————————————————— Specify Matlab outputs —————————————————————————————————— */
	double *RMSEcv;
	plhs[0] = mxCreateDoubleMatrix(n + 1, A, mxREAL);
	RMSEcv = mxGetPr(plhs[0]);

	/* ———— Allocate space for variables that are used multiple times and can be overwritten ——— */
	IsVal     = (bool*)malloc(sizeof(bool)     * m                  ); // [m-by-1]
	Xtrain    = (double*)malloc(sizeof(double) * m    *   n         ); // [m-by-n] (overallocated)
	Xval      = (double*)malloc(sizeof(double) * m    *   n         ); // [m-by-n] (overallocated)
	ytrain    = (double*)malloc(sizeof(double) * m                  ); // [m-by-1] (overallocated)
	yval      = (double*)malloc(sizeof(double) * m                  ); // [m-by-1] (overallocated)
	beta      = (double*)malloc(sizeof(double) * n    *   A         ); // [n-by-A]
	yhat      = (double*)malloc(sizeof(double) * m    *   A         ); // [m-by-A] (overallocated)
	B         = (double*)malloc(sizeof(double) * A    *   2         ); // [A-by-2]
	w         = (double*)malloc(sizeof(double) * n                  ); // [n-by-1]
	wn        = (double*)malloc(sizeof(double)                      ); // [1-by-1]
	W         = (double*)malloc(sizeof(double) * n    * A           ); // [n-by-A]
    rho       = (double*)malloc(sizeof(double)                      ); // [1-by-1]
	rhoi      = (double*)malloc(sizeof(double)                      ); // [1-by-1]
	d         = (double*)malloc(sizeof(double) * n                  ); // [n-by-1]
	tty       = (double*)malloc(sizeof(double)                      ); // [1-by-1]
	Xt        = (double*)malloc(sizeof(double) * n                  ); // [n-by-1]
	Ww        = (double*)malloc(sizeof(double) * A                  ); // [A-by-1] (overallocated)
	WWw       = (double*)malloc(sizeof(double) * n                  ); // [n-by-1] 
	theta     = (double*)malloc(sizeof(double)                      ); // [1-by-1]
	thetai    = (double*)malloc(sizeof(double)                      ); // [1-by-1]
	Tt        = (double*)malloc(sizeof(double) * A                  ); // [A-by-1] (overallocated)

	
	/* Before starting set the seed of the RNG to the number of clock cycles since start.		 */
	srand(clock());

	/* CV-loop starts here																	     */
	for (cvIter = 0; cvIter < MaxIters; cvIter++) {

		/* ————————————— Mark rows of X as either validation rows or training rows ————————————— */
		MarkAsVal(IsVal, NoValObs, m, &valm, &trainm, y);

		/* —————————— Extract data from X and y and place in Xtrain/Xval/ytrain/yval ——————————— */
		ExtractXandY(X, y, Xtrain, Xval, ytrain, yval, m, n, IsVal);

		/* ————————————————————————————————————————————————————————————————————————————————————— */
		/* Shaving step 1. Evaluate matrix with all variables included.							 */
		/* ————————— Call PLS function to estimate Beta using Xtrain & ytrain —————————————————— */
		PLS( Xtrain, ytrain, A, trainm, n, p, beta, B, w, wn, W, rho, rhoi, d, tty, Xt, Ww, WWw, theta, thetai, Tt);
		/* ——————— Use beta to estimate error when predicting yval using Xval —————————————————— */
		Pred( Xval, beta, yval, n, valm, A, yhat, RMSEcv, cvIter, MaxIters, 0);

		/* ————————————————————————————————————————————————————————————————————————————————————— */
		/* Shaving step 2. Evaluate matrix with all except the last variable included.			 */
		/* ————————— Call PLS function to estimate Beta using Xtrain & ytrain —————————————————— */
		PLS(Xtrain, ytrain, A, trainm, n - 1, p, beta, B, w, wn, W, rho, rhoi, d, tty, Xt, Ww, WWw, theta, thetai, Tt);
		/* ——————— Use beta to estimate error when predicting yval using Xval —————————————————— */
		Pred(Xval, beta, yval, n - 1, valm, A, yhat, RMSEcv, cvIter, MaxIters, n);
		/* ————————————————————————————————————————————————————————————————————————————————————— */

		/* ————————————————————————————————————————————————————————————————————————————————————— */
		/* Shaving step 3. Keep last column inactive and swap it one by one with cols 0 : n-1    */
		for (ColToShave = 0; ColToShave < (n - 1); ColToShave++) {

			/* Place ColToShave at the end, effectecly inactivating it.							 */
			SwapCols(Xtrain, Xval, trainm, valm, n, ColToShave + 1);
			/* ———————— Call PLS function to estimate Beta using Xtrain & ytrain ——————————————— */
			PLS(Xtrain, ytrain, A, trainm, n - 1, p, beta, B, w, wn, W, rho, rhoi, d, tty, Xt, Ww, WWw, theta, thetai, Tt);
			/* —————— Use beta to estimate error when predicting yval using Xval ——————————————— */
			Pred(Xval, beta, yval, n - 1, valm, A, yhat, RMSEcv, cvIter, MaxIters, ColToShave + 1);
		}
	}
	
	/* —————————————————————————————————— Free allocated memory ———————————————————————————————— */
	free(IsVal);
	free(Xtrain);
	free(Xval);
	free(ytrain);
	free(yval);
	free(beta);
	free(yhat);
	free(B);
	free(w);
	free(wn);
	free(W);
	free(rho);
	free(rhoi);
	free(d);
	free(tty);
	free(Xt);
	free(Ww);
	free(WWw);
	free(theta);
	free(thetai);
	free(Tt);
}

/* ————————————————————————————————————————————————————————————————————————————————————————————— */
/* Function for estimating Beta using the bidag2 algorithm										 */
void PLS(const double *X, const double *y, int A, size_t m, size_t n, size_t p, double *beta,
	double *B, double *w, double *wn, double *W, double *rho, double *rhoi, double *d,
	double *tty, double *Xt, double *Ww, double *WWw, double *theta, double *thetai, double *Tt){
	
	/* ——————————————————————————  Variable type and name declaration —————————————————————————— */
	double *t;
	double *T;
	double *Xw;
	double *TTt;

	int a, row;
	size_t st_a;

	mwSignedIndex IntConstOne = 1;
	double one = 1.0, zero = 0.0, none = -1.0;

	/* ————————————————— Allocate memory for variables used in the calculation ————————————————— */
	t      = (double*)malloc(sizeof(double)  * m                  ); // [m-by-1]
	T      = (double*)malloc(sizeof(double)  * m        * A       ); // [m-by-A]
	Xw     = (double*)malloc(sizeof(double)  * m                  ); // [m-by-1]
	TTt    = (double*)malloc(sizeof(double)  * m                  ); // [m-by-1]

	
	/* ————————————————————————————————————————————————————————————————————————————————————————— */
	/* ——————————————————————————————— Start of calculation part ——————————————————————————————— */
	/* ——————————————————————————————————————————————————————————————————————————————————————————
	Step 1: w = X'*y; 																             */
	/// X         = [m-by-n]
	/// y         = [m-by-1]
	/// w         = [n-by-1]

	// Matrix-Vector multiplication -> dgemv.
	dgemv("T", &m, &n, &one, X, &m, y, &IntConstOne, &zero, w,&IntConstOne);

	/*—————————————————————————————————————————————————————————————————————————————————————————————
	Step 2: w = w / norm(w); 																     */
	/// w         = [n-by-1]
	/// wn        = [1-by-1]
	
	// Vector norm -> dnrm2.
	*wn = dnrm2(&n, w, &IntConstOne);
	// Inverse resulting scalar to enable multiplication instead of division.
	*wn = 1 / (*wn);
	// Vector-scalar multiplication -> dscal.
	dscal(&n, wn, w, &IntConstOne);

	/*—————————————————————————————————————————————————————————————————————————————————————————————
	Step 3: W(:,CurrentComp) = w;																 */
	/// W         = [n-by-A]
	/// w         = [n-by-1]

	// Copy n doubles from w to W.
	memcpy(W, w, sizeof(double) * n);

	/*—————————————————————————————————————————————————————————————————————————————————————————————
	Step 4: t = X*w;																             */
	/// X         = [m-by-n]
	/// w         = [n-by-1]
	/// t         = [m-by-1]

	// Matrix-Vector multiplication -> dgemv.
	dgemv("N", &m, &n, &one, X, &m, w, &IntConstOne, &zero, t, &IntConstOne);

	/*—————————————————————————————————————————————————————————————————————————————————————————————
	Step 5: rho = norm(t);																         */
	/// t         = [m-by-1]           
	/// rho       = [1-by-1]

	// Vector norm->dnrm2.
	*rho = dnrm2(&m, t, &IntConstOne);

	/*—————————————————————————————————————————————————————————————————————————————————————————————
	Step 6: t = t / rho;																         */
	/// t         = [m-by-1]           
	/// rho       = [1-by-1]
	/// rhoi      = [1-by-1]

	// Create inverse rho (rhoi) scalar to enable multiplication instead of division.
	*rhoi = 1 / (*rho);
	// Vector-scalar multiplication ->dscal.
	dscal(&m, rhoi, t, &IntConstOne);

	/*—————————————————————————————————————————————————————————————————————————————————————————————
	Step 7: T(:,CurrentComp) = t;														         */
	/// T         = [m-by-A]        
	/// t         = [m-by-p]      

	// Copy m doubles from t to T.
	memcpy(T, t, sizeof(double) * m);
	
	/*—————————————————————————————————————————————————————————————————————————————————————————————
	Step 8: B(1, 1) = rho;														                 */
	/// B         = [A-by-2]        
	/// rho       = [1-by-1]      
	
	B[0] = *rho;

	/*—————————————————————————————————————————————————————————————————————————————————————————————
	Step 9: d = w / rho;													                     */
	/// w         = [n-by-1]      
	/// rho       = [1-by-1]
	/// rhoi      = [1-by-1]  
	/// d         = [n-by-1]  

	// Make d = w
	memcpy(d, w, sizeof(double) * n);

	// Then scale it with the inversed version of rho to make it w/rho.
	dscal(&n, rhoi, d, &IntConstOne);

	/*—————————————————————————————————————————————————————————————————————————————————————————————
	Step 9: beta(:, 1) = (t'*y)*d;													             */
	/// t         = [m-by-1]      
	/// d         = [n-by-1]  
	/// y         = [m-by-1]
	/// tty       = [1-by-1]
	/// beta(:,1) = [n-by-1]

	// (t'*y) vector-vector product -> ddot.
	*tty = ddot(&m, t, &IntConstOne, y, &IntConstOne);

	// (d*tty) vector-scalar product. (Here treated as matrix-matrix).
	dgemm("N", "N", &n, &p, &p, &one, d, &n, tty, &p, &zero, beta, &n);
	
	//* Start of component loop.															     */
	for (a = 1; a < A; a++) {
		/* size_t version of a (preferred by BLAS instead of int). 								 */
		st_a = (size_t)a;

		/*—————————————————————————————————————————————————————————————————————————————————————————
		Step 10: w = X'*t - rho*w;													             */
		/// w         = [n-by-1]      
		/// rho       = [1-by-1]      
		/// X         = [m-by-n]
		/// t         = [m-by-1]      
		/// Xt        = [n-by-1]

		// Xt = X'*t matrix-vector multiplication -> dgemv.
		dgemv("T", &m, &n, &one, X, &m, t, &IntConstOne, &zero, Xt, &IntConstOne);
		
		// The value of rho wont be used again, overwrite with -1*rho.
		*rho = -1 * (*rho);

		// w = rho*w scalar-vector multiplication -> dscal.
		dscal(&n, rho, w, &IntConstOne);

		// w = 1*Xt + w vector times constant plus vector -> daxpy.
		daxpy(&n, &one, Xt, &IntConstOne, w, &IntConstOne);

	
		/*—————————————————————————————————————————————————————————————————————————————————————————
		Step 11: w = w - W(:,1:a)*(W(:,1:a)'*w);											     */
		/// W(:,1:a)   = [n-by-a]
		/// w          = [n-by-1]      
		/// Ww         = [a-by-1]      
		/// WWw        = [n-by-1]

		// Ww = (W(:,1:a)'*w) matrix-vector multiplication -> dgemm.
		dgemv("T", &n, &st_a, &one, W, &n, w, &IntConstOne, &zero, Ww, &IntConstOne);

		// WWw = W(:,1:a)*(W(:,1:a)'*w)  matrix-vector multiplication -> dgemm.
		dgemv("N", &n, &st_a, &one, W, &n, Ww, &IntConstOne, &zero, WWw, &IntConstOne);

		//  w ←  w + -1 * WWw constant times a vector plus a vector -> daxpy.
		daxpy(&n, &none, WWw, &IntConstOne, w, &IntConstOne);

		/*—————————————————————————————————————————————————————————————————————————————————————————
		Step 12: theta = norm(w);													             */
		/// w         = [n-by-1]      
		/// theta     = [1-by-1]      

		// Vector norm -> dnrm2.
		*theta = dnrm2(&n, w, &IntConstOne);

		/*—————————————————————————————————————————————————————————————————————————————————————————
		Step 13: w = w / theta;													                 */
		/// w         = [n-by-1]      
		/// theta     = [1-by-1]     

		// w = w * (1/theta) scalar-vector multiplication -> dscal.
		*thetai = 1 / (*theta);
		dscal(&n, thetai, w, &IntConstOne);

		/*—————————————————————————————————————————————————————————————————————————————————————————
		Step 14: W(:,a) = w;													                 */
		/// w         = [n-by-1]      
		/// W         = [n-by-A]     
		
		// Copy n elemments from w into W at offset n * a.
		memcpy(W + n * a, w, sizeof(double) * n);

		/*—————————————————————————————————————————————————————————————————————————————————————————
		Step 15: t = X*w - theta*t;													             */
		/// X         = [m-by-n]      
		/// w         = [n-by-1]      
		/// theta     = [1-by-1]     
		/// t         = [m-by-1]     
		/// Xw        = [m-by-1]

		// Xw = X*w = matrix vector multiplication -> dgemv.
		dgemv("N", &m, &n, &one, X, &m, w, &IntConstOne, &zero, Xw, &IntConstOne);

		// t = Xw - theta*t Better to solve with daxpy?
		//for (row = 0; row < m; row++){
	    //	t[row] = Xw[row] - (*theta) * t[row];
		//}
		dscal(&m, theta, t, &IntConstOne);
		dscal(&m, &none, t, &IntConstOne);
		daxpy(&m, &one, Xw, &IntConstOne, t, &IntConstOne);

		/*—————————————————————————————————————————————————————————————————————————————————————————
		Step 16: t = t - T(:,1:a)*(T(:,1:a)'*t);												 */   
		/// t         = [m-by-1]     
		/// T(:,1:a)  = [m-by-a]
		/// Tt        = [a-by-1]
		/// TTt       = [m-by-1]

		// Tt = (T(:,1:a)'*t) matrix-vector multiplication -> dgemv. 
		dgemv("T", &m, &st_a, &one, T, &m, t, &IntConstOne, &zero, Tt, &IntConstOne);

		// TTt = T(:,1:a)*(T(:,1:a)'*t) matrix-vector multiplication -> dgemv.
		dgemv("N", &m, &st_a, &one, T, &m, Tt, &IntConstOne, &zero, TTt, &IntConstOne);

		//  t ←  t + -1 * TTt constant times a vector plus a vector -> daxpy.
		daxpy(&m, &none, TTt, &IntConstOne, t, &IntConstOne);

		/*—————————————————————————————————————————————————————————————————————————————————————————
		Step 17: rho = norm(t);													                 */
		/// t         = [m-by-1]     
		/// rho       = [1-by-1]

		// Vector norm -> dnrm2.
		*rho = dnrm2(&m, t, &IntConstOne);
		
		/*—————————————————————————————————————————————————————————————————————————————————————————
		Step 18: t = t/rho;													                     */
		/// t         = [m-by-1]     
		/// rho       = [1-by-1]
		
		// Update inverse version of rho.
		*rhoi = 1 / (*rho);

		// t = t * (1/rho) scalar-vector multiplication -> dscal
		dscal(&m, rhoi, t, &IntConstOne);

		/*—————————————————————————————————————————————————————————————————————————————————————————
		Step 19: T(:, a) = t;													                 */
		/// T         = [m-by-A]     
		/// t         = [m-by-1]   

		// Copy m elemments from t into T at offset m * a
		memcpy(T + m * a, t, sizeof(double) * m);

		/*—————————————————————————————————————————————————————————————————————————————————————————
		Step 20: B(a, 1) = rho;													                 */
		/// B         = [A-by-2]     
		/// rho       = [1-by-1]   
	
		B[a] = *rho;

		/*—————————————————————————————————————————————————————————————————————————————————————————
		Step 21: B(a - 1, 2) = theta;													         */
		/// B         = [A-by-2]     
		/// theta     = [1-by-1]   

		B[(a-1) + A] = *theta;

		/*—————————————————————————————————————————————————————————————————————————————————————————
		Step 22: d = (w - theta*d) / rho;													     */
		/// d         = [n-by-1]  
		/// rho       = [1-by-1]   
		/// theta     = [1-by-1]   
		/// w         = [n-by-1] 
	
		for (row = 0; row < n; row++){
			 d[row] = ( w[row] - (*theta) * d[row] ) / (*rho);
		}

		/*—————————————————————————————————————————————————————————————————————————————————————————
		Step 23: beta(:, a) = beta(:, a - 1) + (t'*y)*d;									     */
		/// beta      = [n-by-A]  
		/// t         = [m-by-1]   
		/// y         = [m-by-1]   
		/// d         = [n-by-1] 

		// (t'*y) vector-vector product -> ddot
		*tty = ddot(&m, t, &IntConstOne, y, &IntConstOne);
		
		// (tty*d) = scalar-vector product. (Here treated as matrix-matrix).
		dgemm("N", "N", &n, &p, &p, &one, d, &n, tty, &p, &zero, beta + n * a , &n);

		//  beta(:, a) ←  beta(:, a) + 1 * beta(:, a - 1) -> daxpy
		daxpy(&n, &one, beta + n * (a-1), &IntConstOne, beta + n * a, &IntConstOne);

		//for (row = 0; row < n; row++) {
			//beta[row + n * a] = beta[row + n * a] + beta[row + n * (a - 1)];
			//beta[row + n * a] = beta[row + n * (a - 1)];
		//}

    /* End of component loop.																	 */
	}

	free(t);
	free(T);
	free(Xw);
	free(TTt);
}
/* ————————————————————————————————————————————————————————————————————————————————————————————— */

/* ————————————————————————————————————————————————————————————————————————————————————————————— */
/* Function for drawing a random integer that lies within range.								 */
int randr(unsigned int min, unsigned int max) {
	return min + rand() / (RAND_MAX / (max - min + 1) + 1);
}
/* ————————————————————————————————————————————————————————————————————————————————————————————— */

/* ————————————————————————————————————————————————————————————————————————————————————————————— */
/* Function for marking rows in the data as validation or not during CV							 */
void MarkAsVal(bool *IsVal, int NoValObs, size_t m, size_t *valm, size_t *trainm, const double *y) {
	int activeobs, row, index;
	bool alreadyactive;

	/* Initialize all IsVal elements to false */
	for (row = 0; row < m; row++) {
		IsVal[row] = false;
	}

	activeobs = 0;
	while (activeobs < NoValObs) { // NOTE! this needs a check aginst input NoValObs larger than unique(y)! 
								   /* Draw random rownumber between 0 and m */
		index = randr(0, m);
		alreadyactive = false;
		/* Mark all the rows with the same response as validation to keep replicates together */
		for (row = 0; row < m; row++) {
			if (y[row] == y[index]) {
				/* If IsVal was 0 it means this is a newly found observation. */
				if (IsVal[row] == false & alreadyactive == false) {
					activeobs += 1;
					alreadyactive = true;
				}
				/* If IsVal is already 1 stop looping and pick a new observation to save time */
				else if (IsVal[row] == true) {
					break;
				}
				IsVal[row] = true;
			}
		}
	}

	/* Calculate the number of rows in Xval/yval and Xtrain/ytrain so they can be allocated */
	*valm = 0;
	for (row = 0; row < m; row++) {
		if (IsVal[row] == true) {
			*valm += 1;
		}
	}
	*trainm = m - (*valm);
}
/* ————————————————————————————————————————————————————————————————————————————————————————————— */

/* ————————————————————————————————————————————————————————————————————————————————————————————— */
/* Function for populating Xtrain Xval ytrain yval given known IsVal vector					     */
void ExtractXandY(const double *X, const double *y, double *Xtrain, double *Xval, double *ytrain,
	double *yval, size_t m, size_t n, bool *IsVal) {
	
	int col, row, xte, xve, yve, yte;

	xte = xve = 0;
	yte = yve = 0;

	/* Outer loop over all columns of X */
	for (col = 0; col < n; col++) {
		/* Inner loop over all rows of X */
		for (row = 0; row < m; row++) {

			if (IsVal[row] == true) {
				Xval[xve] = X[row + col * m];
				xve += 1;
				if (col == 0) {
					yval[yve] = y[row];
					yve += 1;
				}
			}
			else {
				Xtrain[xte] = X[row + col * m];
				xte += 1;
				if (col == 0) {
					ytrain[yte] = y[row];
					yte += 1;
				}
			}
		}
	}
}
/* ————————————————————————————————————————————————————————————————————————————————————————————— */

/* ————————————————————————————————————————————————————————————————————————————————————————————— */
/* Function for calculating validation error using Xval and beta								 */
void Pred(double *Xval, double *beta, double *yval, size_t n, size_t valm, int A, double *yhat, double *RMSEcv, int cvIter, int MaxIters, int ShavingIndex) {

	size_t st_A;
	int col;
	double one = 1.0, zero = 0.0, none = -1.0;
	mwSignedIndex IntConstOne = 1;
	st_A = (size_t)A;

	/* yhat = Xval*beta														                     */
	dgemm("N", "N", &valm, &st_A, &n, &one, Xval, &valm, beta, &n, &zero, yhat, &valm);

	/* Loop the columns of yhat and calulate the error for each component.						 */
	for (col = 0; col < A; col++) {

		/* yhat is here overwritten and becomes epsilon	where epsilon = yhat + (-Yval).	         */
		daxpy(&valm, &none, yval, &IntConstOne, yhat + col * valm, &IntConstOne);

		/* Calculate the Mean Squared Error by taking the dot product of each error column with
		itself and then dividing by the number of observations (valm). At the same time, add it
		to the	existing value of the RMSE vector, thereby performing a summation of the errors
		of each cross-validation iteration at the same time (prevents the need of a 2D matrix).	 */
		if (ShavingIndex == 0) {
			RMSEcv[ShavingIndex + col*(n + 1)] += ddot(&valm, yhat + col * valm, &IntConstOne, yhat + col * valm, &IntConstOne) / valm;
		}
		else {
			RMSEcv[ShavingIndex + col*(n + 2)] += ddot(&valm, yhat + col * valm, &IntConstOne, yhat + col * valm, &IntConstOne) / valm;
		}
		/* If its the last cross-validation iteration, finalize RMSEcv (which right now
		represents sum(MSE,1) [folds-by-components] ) by dividing by the number of cv iterations
		and taking the square-root of the currently squared errors.								 */
		if (cvIter == MaxIters - 1) {
			if (ShavingIndex == 0) {
				RMSEcv[ShavingIndex + col*(n + 1)] = sqrt(RMSEcv[ShavingIndex + col*(n + 1)] / MaxIters);
			}
			else {
				RMSEcv[ShavingIndex + col*(n + 2)] = sqrt(RMSEcv[ShavingIndex + col*(n + 2)] / MaxIters);
			}
		}

	}

}
/* ————————————————————————————————————————————————————————————————————————————————————————————— */

/* ————————————————————————————————————————————————————————————————————————————————————————————— */
/* Function for placing a column at the and of a matrix (swapping it with the last column)       */
void SwapCols(double *Xtrain, double *Xval, size_t trainm, size_t valm, size_t n, int ColToShave) {

	mwSignedIndex IntConstOne = 1;

	/* Use dswap to swap location of two column vectors in Xtrain.								 */
	dswap(&trainm, Xtrain + (n - 1) * trainm, &IntConstOne,	Xtrain + ColToShave * trainm, &IntConstOne);

	/* Use dswap to swap location of two column vectors in Xval.								 */
	dswap(&valm, Xval + (n - 1) * valm, &IntConstOne, Xval + ColToShave * valm, &IntConstOne);
}
/* ————————————————————————————————————————————————————————————————————————————————————————————— */