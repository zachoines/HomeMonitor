# To Install on rasberry pi
* Overview
	* To develope an AI powered target acquisition and tracking system for use on mobile and IOT based camera systems.

* Technologies
	* Target acquisition performed by openCV C++ API. 
		* More advance features pinned for future utilizing Pytorch.
	* Target Tracking with PID's 
	* Advance PID autotuning via Soft Actor Critic Reinforcement learning

* Install
	* Opencv4 c++
		 * [Here](https://cv-tricks.com/installation/opencv-4-1-ubuntu18-04/) is an excellent walkthrough of the process.

	* Pytorch c++
		* Install from source
			* pip3 install -U pip3
			* git clone http://github.com/pytorch/pytorch
			* cd pytorch
			* git submodule update --init
			* sudo pip3 install -U setuptools
			* sudo pip3 install -r requirements.txt
			* python3 setup.py build
			* sudo python setup.py develop
		* Example g++ compiler and linker args (changes depending on where you installed Pytorch)
			* export LD_LIBRARY_PATH=/home/pi/pytorch/build/lib:$LD_LIBRARY_PATH 
			* g++ -std=c++17 -o test *.cpp 
				-I/home/pi/pytorch/build/lib.linux-armv7l-3.7/torch/include 
				-I/home/pi/pytorch/build/lib.linux-armv7l-3.7/torch/include/torch/csrc 
				-I/home/pi/pytorch/build/lib.linux-armv7l-3.7/torch/include/torch/csrc/api/include 
				-L/home/pi/pytorch/build/lib 
				-lc10 -ltorch -ltorch_cpu
			* ./test

	* lib Boost
		* sudo apt-get install libboost
		* include dir as /usr/include/boost
		* import libs: 

	* Servo controller lib for RBPI4
		* I use this [PCA965](https://github.com/Reinbert/pca9685) lib

