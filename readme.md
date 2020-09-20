# Overview
* Goal
	* To develope an AI powered target acquisition and tracking system for use on mobile and IOT based camera systems.

* Technologies
	* Target acquisition performed by OpenCV 4 C++ API. Currently face detection supported.
		* More advance features pinned for future utilizing Pytorch C++ API.
	* Target tracking with PID's and servo control
	* Advance PID autotuning via Soft Actor Critic Reinforcement learning

# To Install on rasberry pi
* Recommended hardware
	* [Raspberry Pi 4 Model B](https://www.raspberrypi.org/products/raspberry-pi-4-model-b/)
	* [Camera Module V2](https://www.raspberrypi.org/products/camera-module-v2/)
	* 2x High speed servos. Ideally, those with metal gear and low amperage.
	* Pan-tilt servo mounts, [like these](https://www.servocity.com/pan-tilt-kits/).
* Recommend making Python 3 env first 
	* export WORKON_HOME=$HOME/.virtualenvs
	* export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
	* export VIRTUALENVWRAPPER_VIRTUALENV=/usr/local/bin/virtualenv
	* source /usr/local/bin/virtualenvwrapper.sh
	* export VIRTUALENVWRAPPER_ENV_BIN_DIR=bin
	* mkvirtualenv build

* OpenCV 4 c++
	* [Here](https://cv-tricks.com/installation/opencv-4-1-ubuntu18-04/) is an excellent walkthrough of the process.

* Pytorch c++
	* Install from source
		* cd /home/pi/
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
	* Sudo apt-get install libboost
	* Include dir via linker command: -I/usr/include/boost

* Servo controller lib for RBPI4
	* I use this [PCA965](https://github.com/Reinbert/pca9685) lib

