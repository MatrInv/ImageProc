#ImageProc

Image processor that performs DENOISING, EDGE DETECTION, IMAGE SEGMENTATION as well as MORPHOLOGICAL OPERATIONS using structuring elements such as dilatation, erosion, opening and closing.

The program has been written with C++11 and OpenCV4.5 . So, be aware that the program may be not working with another versions of these two.

To compile the main.cpp file, you must first configure the compilation with CMakeLists.txt :

	cd <ImageProc directory>
	cmake -DOpenCV_DIR=<opencv-root>/lib/cmake/opencv4 ./
	make

<opencv-root> is usually usr/local (in linux systems).

In order to run the executable :

	chmod +x ImageProc
	
	ImageProc <path-to-an-image> <operation>

<operation> is an number refering to an operation in the correlation table below :

	(1)   Dilatation
	(2)   Erosion
	(3)   Opening
	(4)   Closing
	(5)   Denoising
	(6)   Internal gradient
	(7)   External gradient
	(8)   Morphological gradient
	(9)   Regional minima segmentation
	(10)  Watershed algorithm

1, 2, 3, 4 are morphological operations used in all the other features of the program.
5 is used to remove salt&pepper noise.
6, 7, 8 are gradients used for edge detection.
9, 10 are algorithmes used for image segmentation. The different regions are shown in different grayscales.

You can use the directory img_test to test these features.
I would propose to you the examples below :

	#Denoising
	ImageProc ./img_test/bottle_bruit.png 5
	ImageProc ./img_test/tree_bruit.png 5

	#edge segmentation
	ImageProc ./img_test/tree.png 9
	ImageProc ./img_test/daisy.png 9
	ImageProc ./img_test/lotus.png 9

	#image segmentation
	ImageProc ./img_test/tree.png 10
	ImageProc ./img_test/daisy.png 10
	ImageProc ./img_test/lotus.png 10

You can find the result of all the processed images in the zip archive "res.zip".

By the way, at the end of the execution, the program asks you if you want to save the processed image in the current directory as "img_res.png". If you want so, press the S key, else press any other key.
