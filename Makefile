all:
	python setup.py build_ext --inplace
	cython -a fast_seg.pyx
