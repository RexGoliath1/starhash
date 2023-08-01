FROM ubuntu:latest
ARG DEBIAN_FRONTEND=noninteractive

# Install.
# sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list && \
RUN \
	apt-get update && \
	apt-get -y upgrade && \
	apt-get install -y build-essential cmake bear git gdb unzip pkg-config && \
	apt-get install -y libc-dev libffi-dev && \
	apt-get install -y libjpeg-dev libpng-dev && \
	apt-get install -y libavcodec-dev libavformat-dev libswscale-dev && \
	apt-get install -y libgtk2.0-dev libcanberra-gtk* libgtk-3-dev && \
	apt-get install -y libgstreamer1.0-dev gstreamer1.0-gtk3 && \
	apt-get install -y libgstreamer-plugins-base1.0-dev gstreamer1.0-gl && \
	apt-get install -y libxvidcore-dev libx264-dev && \
	apt-get install -y python3-dev python3-numpy python3-pip python3-venv && \
	apt-get install -y python3-tk python-is-python3 && \ 
	apt-get install -y libtbb2 libtbb-dev libdc1394-dev && \
	apt-get install -y libv4l-dev v4l-utils && \
	apt-get install -y libopenblas-dev libatlas-base-dev libblas-dev && \
	apt-get install -y liblapack-dev gfortran libhdf5-dev && \
	apt-get install -y libprotobuf-dev libgoogle-glog-dev libgflags-dev && \
	apt-get install -y protobuf-compiler ffmpeg && \
	apt-get install -y software-properties-common && \
	apt-get install -y byobu curl git htop man unzip vim wget && \
	apt-get install -y libhdf5-dev && \
	apt-get install -y zsh dconf-cli && \
	apt-get install -y ripgrep golang-go nodejs npm && \
	apt-get install -y language-pack-en && \
	update-locale && \
	rm -rf /var/lib/apt/lists/*
RUN apt-get install -y locales
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

RUN apt update
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
RUN apt install fonts-powerline
RUN chsh -s /usr/bin/zsh
RUN git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
RUN git clone https://github.com/zsh-users/zsh-autosuggestions.git $ZSH_CUSTOM/plugins/zsh-autosuggestions 
RUN git clone https://github.com/zsh-users/zsh-syntax-highlighting.git $ZSH_CUSTOM/plugins/zsh-syntax-highlighting


#  Build neovim (this is probably incomplete. Need to build zsh autosuggestions and configure powerline)
RUN apt-get install -y ninja-build gettext && \
	git clone https://github.com/neovim/neovim && \
	cd neovim && make CMAKE_BUILD_TYPE=RelWithDebInfo && \
	make install

# Lazygit install
RUN LAZYGIT_VERSION=$(curl -s "https://api.github.com/repos/jesseduffield/lazygit/releases/latest" | grep -Po '"tag_name": "v\K[^"]*') && \
	curl -Lo lazygit.tar.gz "https://github.com/jesseduffield/lazygit/releases/latest/download/lazygit_${LAZYGIT_VERSION}_Linux_x86_64.tar.gz" && \
	tar xf lazygit.tar.gz lazygit && \
	install lazygit /usr/local/bin

# A few additions to end of .zshrc
RUN echo 'alias vim="nvim"' >> ~/.zshrc && \
	echo 'alias gs="git status"' >> ~/.zshrc

# Install astrovim dependancies and astrovim
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
RUN echo 'source $HOME/.cargo/env' >> $HOME/.zshrc
ENV PATH="/root/.cargo/bin:${PATH}"


RUN cargo install tree-sitter-cli && \
	cargo install bottom --locked && \
	git clone --depth 1 https://github.com/AstroNvim/AstroNvim ~/.config/nvim

# Set up python virtualenv
RUN pip install virtualenvwrapper && \
	echo "export WORKON_HOME=~/Envs" >> ~/.zshrc && \
	export WORKON_HOME=~/Envs && \
	mkdir -p $WORKON_HOME 
RUN /bin/bash /usr/local/bin/virtualenvwrapper.sh && \
	echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.zshrc && \
	pip install opencv-python-headless matplotlib scipy pandas numpy cython astropy lxml sphinx spiceypy astroquery dill psutil

# WARNING: c-compiler and openmp doesn't work on M1


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

RUN wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip && \
	unzip eigen-3.4.0.zip && \
	mv eigen-3.4.0 eigen && \
	rm eigen-3.4.0.zip

RUN cd ~/eigen && \
	mkdir build && \
	cd build && \
	cmake .. && \
	make && \
	make install && \
	ldconfig

RUN cd ~/ && \
	rm -rf ~/eigen 

RUN wget https://github.com/libigl/libigl/archive/refs/tags/v2.4.0.zip && \
	unzip v2.4.0.zip && \
	mv libigl-2.4.0 libigl && \
	rm v2.4.0.zip
	cd ~/libigl && \
	mkdir build && \
	cd build && \
	cmake .. && \
	make && \
	make install && \
	ldconfig

RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.8.0.zip && \
	wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.8.0.zip && \
	unzip opencv.zip && \
	unzip opencv_contrib.zip && \
	mv opencv-4.8.0 opencv && \
	mv opencv_contrib-4.8.0 opencv_contrib && \
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

RUN pip install numpy && \
	pip install matplotlib && \
	pip install opencv-python && \
	pip install opencv-contrib-python && \
	pip install astroquery && \
	pip install pandas && \
	pip install astropy && \
	pip install scipy

# Update locales for time prop
RUN apt-get update && apt-get install -y locales && \
	locale-gen en_US.UTF-8 && \
	export LANG=en_US.UTF-8 && \
	echo "export LANG=en_US.UTF-8" >> ~/.bashrc && \
	echo "export LANG=en_US.UTF-8" >> ~/.zshrc

# Download a few useful repos 

RUN echo "Done. Some things that need doing ..." && \
	echo "plugins (git zsh-autosuggestions zsh-syntax-highlighting)" && \
	echo "ZSH_THEME=agnoster. No clue how i did dracula stuff" && \
	echo "ZSH_THEME="powerlevel10k/powerlevel10k"" && \
	echo "p10k-configure" && \
	echo "git clone git@github.com:nasa/COTS-Star-Tracker.git" && \
	echo "git clone git@github.com:RexGoliath1/starhash.git" 
