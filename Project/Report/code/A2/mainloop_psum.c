int main(){
	int i;
	int a[10] = {1, 2, 4, 8, 12, 16, 20, 24, 28, 32};
	for (i = 0; i < 10; i++){
		omp_set_num_threads(a[i]);
		func();
	}
	return 0;
}
