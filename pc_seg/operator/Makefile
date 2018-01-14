all: test_compute_dists

test_compute_dists: compute_dists.o
	g++ test_compute_dists.cpp compute_dists.o -o test_compute_dists -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -lcudart

compute_dists.o:
	nvcc -c compute_dists.cu

clean:
	rm -rf *.o test_compute_dists
