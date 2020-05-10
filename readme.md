# TO Install on rasberry pi
* Install opencv4
	 * [Here](https://cv-tricks.com/installation/opencv-4-1-ubuntu18-04/) is an excellent walkthrough of the process.
* Set camera that loads on startup if using reasberry pi camera
	* Open modile loaded on boot: sudo nano /etc/modules
	* Enter new module: bcm2835-v4l2
