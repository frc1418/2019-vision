set -e

apt-get update -y
apt-get upgrade -y
apt-get install -y build-essential cmake pkg-config \
    libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev \
    libgtk2.0-dev libgtk-3-dev \
    libatlas-base-dev gfortran \
    python3-dev python3-pip

cd ~
wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.3.0.zip
unzip opencv.zip
wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.3.0.zip
unzip opencv_contrib.zip

pip3 install virtualenv virtualenv-wrapper
rm -rf ~/.cache/pip
