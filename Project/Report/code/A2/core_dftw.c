int DFT(int idft, double *xr, double *xi, double *Xr_o, double *Xi_o, int N) {
	#pragma omp parallel
	{	
		#pragma omp for schedule(guided)
		for (int k = 0; k < N; k++) {
			double re = 0.0;
			double im = 0.0;
			for (int n = 0; n < N; n++) {
				// Real part of X[k]
				re += xr[n] * cos(n * k * PI2 / N) + idft * xi[n] * sin(n * k * PI2 / N);

				// Imaginary part of X[k]
				im += -idft * xr[n] * sin(n * k * PI2 / N) + xi[n] * cos(n * k * PI2 / N);
			}
			Xr_o[k] += re;
			Xi_o[k] += im;
		}
	}

	// normalize if you are doing IDFT
	if (idft == -1) {
		for (int n = 0; n < N; n++) {
			Xr_o[n] /= N;
			Xi_o[n] /= N;
		}
	}
	return 1;
}