FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive

# Install.
# sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list && \
RUN \
  apt-get update && \
  apt-get -y upgrade && \
	apt-get install -y build-essential cmake git gdb unzip pkg-config && \
	apt-get install -y libjpeg-dev libpng-dev && \
	apt-get install -y libavcodec-dev libavformat-dev libswscale-dev && \
	apt-get install -y libgtk2.0-dev libcanberra-gtk* libgtk-3-dev && \
	apt-get install -y libgstreamer1.0-dev gstreamer1.0-gtk3 && \
	apt-get install -y libgstreamer-plugins-base1.0-dev gstreamer1.0-gl && \
	apt-get install -y libxvidcore-dev libx264-dev && \
	apt-get install -y python3-dev python3-numpy python3-pip && \
	apt-get install -y libtbb2 libtbb-dev libdc1394-22-dev && \
	apt-get install -y libv4l-dev v4l-utils && \
	apt-get install -y libopenblas-dev libatlas-base-dev libblas-dev && \
	apt-get install -y liblapack-dev gfortran libhdf5-dev && \
	apt-get install -y libprotobuf-dev libgoogle-glog-dev libgflags-dev && \
	apt-get install -y protobuf-compiler ffmpeg && \
  apt-get install -y software-properties-common && \
  apt-get install -y byobu curl git htop man unzip vim wget && \
	apt-get install -y qt5-default && \
	apt-get install -y libhdf5-dev && \
  rm -rf /var/lib/apt/lists/*

# Add files.
# ADD root/.bashrc /root/.bashrc
# ADD root/.gitconfig /root/.gitconfig
# ADD root/.scripts /root/.scripts

# Set environment variables.
ENV HOME /root

# Define working directory.
WORKDIR /root

# Define default command.
CMD ["bash"]

RUN wget -O https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip && \
	unzip eigen-3.4.0.zip && \
	mv eigen-3.4.0.zip && \
	rm eigen-3.4.0.zip

RUN cd ~/eigen && \
	mkdir build && \
	cd build && \
	cmake .. && \
	make && \
	make install && \
	ldconfig && \

RUN cd ~/ && \
	rm -rf ~/eigen 

RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.5.5.zip && \
	wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.5.5.zip && \
	unzip opencv.zip && \
	unzip opencv_contrib.zip && \
	mv opencv-4.5.5 opencv && \
	mv opencv_contrib-4.5.5 opencv_contrib && \
	rm opencv.zip && \
	rm opencv_contrib.zip 

RUN cd ~/opencv && \
	mkdir build && \
	cd build && \
	cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
	-D WITH_OPENMP=ON \
	-D WITH_OPENCL=OFF \
	-D BUILD_TIFF=ON \
	-D WITH_FFMPEG=ON \
	-D WITH_TBB=ON \
	-D BUILD_TBB=ON \
	-D WITH_GSTREAMER=ON \
	-D BUILD_TESTS=OFF \
	-D WITH_EIGEN=OFF \
	-D WITH_V4L=ON \
	-D WITH_LIBV4L=ON \
	-D WITH_VTK=OFF \
	-D WITH_QT=OFF \
	-D OPENCV_ENABLE_NONFREE=ON \
	-D INSTALL_C_EXAMPLES=OFF \
	-D INSTALL_PYTHON_EXAMPLES=OFF \
	-D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
	-D OPENCV_GENERATE_PKGCONFIG=ON \
	-D BUILD_EXAMPLES=OFF .. && \
	make -j$(nproc) && \
	make install && \ 
	ldconfig && \
	apt-get update 

RUN cd ~/ && \
	rm -rf ~/opencv && \
	rm -rf ~/opencv_contrib
