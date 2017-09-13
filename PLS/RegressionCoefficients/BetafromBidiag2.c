/* ————————————————————————————————————————————————————————————————————————————————————————————————
Partial-least-squares regression using the Bidiag2 algorithm.
———————————————————————————————————————————————————————————————————————————————————————————————————
This is a MEX function which uses BLAS to replicate the calculation steps of the Bidiag2 algorithm
as described in:

Björck Å, Indahl. UG. Fast and stable partial least squares modelling:
A benchmark study with theoretical comments. Journal of Chemometrics. 2017;e2898.

The function takes 3 inputs (all should be double precision):
* Input 1: a [m x n] design matrix with observations as rows and variables as columns.
* Input 2: a [m x 1] vector with response values (multiple responses is not supported).
* Input 3: a [1 x 1] scalar 'A' specifying the maximum number of PLS components to calculate.

The function outputs 3 variables:
* Output 1: a [n x A] matrix with regression coefficients.

Example on how to compile and run from Matlab:
% Compile .C to .mexw64
>> mex -largeArrayDims -lmwblas BetafromBidiag2.c

% Run from Matlab when compiled:
>> X = rand(10000, 256);
>> y = rand(10000, 1);
>> A = 10;

>> [ beta ] = BetafromBidiag2( X , y, A );

Example of compatible C compilers:
* Microsoft Visual C++ 2013 Professional (C)
* Microsoft Visual C++ 2015 Professional (C)
* Intel Parallel Studio XE 2017

Written 2017-08-14 by
petter.stefansson@nmbu.no
———————————————————————————————————————————————————————————————————————————————————————————————— */

#include "mex.h"	// needed to communicate with matlab
#include "blas.h"	// needed for blas functions
#include "string.h" // needed to avoid compiler warning due to memcpy when using old compilers

void PLS(const double *X, const double *y, int A, size_t m, size_t n, size_t p, double *beta);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	/* —————————————————————————— Variable type and name declaration ——————————————————————————— */
	const double *X;         // Input 1 (X)
	const double *y;         // Input 2 (Y)
	const double *MC;		 // Input 3 (max comps)
	double *beta;	         // Output 1 (Beta)

	size_t m, n, p;          // Size variables

	/* ———————————————————————— Get pointers from the input variables —————————————————————————— */
	X  = mxGetPr(prhs[0]);   // First input matrix  X.
	y  = mxGetPr(prhs[1]);   // Second input vector Y.
	MC = mxGetPr(prhs[2]);   // Max number of components to calculate.

	/* ——————————————————————— Get the dimensions of the input variables ——————————————————————— */
	m = mxGetM(prhs[0]);     // Number of rows in    X.
	n = mxGetN(prhs[0]);     // Number of columns in X.
	p = mxGetN(prhs[1]);     // Number of columns in Y.

	/* ——————————————————————————— If input A is larger than n, let A = n —————————————————————— */
	double A = *MC;
	if ((A > n) || (A < 1)) { A = n; }

	/* ——————————————————————————————— Specify Matlab outputs —————————————————————————————————— */
	plhs[0] = mxCreateDoubleMatrix(n, A, mxREAL);
	beta = mxGetPr(plhs[0]);

	/* ———————————————————————— Call PLS function to estimate Beta ————————————————————————————— */
	PLS( X, y, (int)A, m, n, p, beta);

}

void PLS(const double *X, const double *y, int A, size_t m, size_t n, size_t p, double *beta){
	
	/* ——————————————————————————  Variable type and name declaration —————————————————————————— */
	double *B;
	double *w;
	double *wn;
	double *W;
	double *t;
	double *rho;
	double *rhoi;
	double *T;
	double *d;
	double *tty;
	double *Xt;
	double *Ww;
	double *WWw;
	double *theta;
	double *thetai;
	double *Xw;
	double *Tt;
	double *TTt;

	int a, row;
	size_t st_a;

	mwSignedIndex IntConstOne = 1;
	double one = 1.0, zero = 0.0, none = -1.0;

	/* ————————————————— Allocate memory for variables used in the calculation ————————————————— */
	B      = (double*)malloc(sizeof(double)  * A        * 2       ); // [A-by-2]
	w      = (double*)malloc(sizeof(double)  * n                  ); // [n-by-1]
	wn     = (double*)malloc(sizeof(double)                       ); // [1-by-1]
	W      = (double*)malloc(sizeof(double)  * n        * A       ); // [n-by-A]
	t      = (double*)malloc(sizeof(double)  * m                  ); // [m-by-1]
	rho    = (double*)malloc(sizeof(double)                       ); // [1-by-1]
	rhoi   = (double*)malloc(sizeof(double)                       ); // [1-by-1]
	T      = (double*)malloc(sizeof(double)  * m        * A       ); // [m-by-A]
	d      = (double*)malloc(sizeof(double)  * n                  ); // [n-by-1]
	tty    = (double*)malloc(sizeof(double)                       ); // [1-by-1]
	Xt     = (double*)malloc(sizeof(double)  * n                  ); // [n-by-1]
	Ww     = (double*)malloc(sizeof(double)  * A                  ); // [A-by-1] (overallocated)
	WWw    = (double*)malloc(sizeof(double)  * n                  ); // [n-by-1] 
	theta  = (double*)malloc(sizeof(double)                       ); // [1-by-1]
	thetai = (double*)malloc(sizeof(double)                       ); // [1-by-1]
	Xw     = (double*)malloc(sizeof(double)  * m                  ); // [m-by-1]
	Tt     = (double*)malloc(sizeof(double)  * A                  ); // [A-by-1] (overallocated)
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

	free(B);
	free(w);
	free(wn);
	free(W);
	free(t);
	free(rho);
	free(rhoi);
	free(T);
	free(d);
	free(tty);
	free(Xt);
	free(Ww);
	free(WWw);
	free(theta);
	free(thetai);
	free(Xw);
	free(Tt);
	free(TTt);
}