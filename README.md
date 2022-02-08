# c_cuda

The development environment used is a wsl Ubuntu 20.04, and
to install the cuda package follow this commands. 

-sudo apt-get install gcc
-wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
-sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
-sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
-sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
-sudo apt-get update
-sudo apt-get -y install cuda
-sudo apt-get install nvidia-cuda-toolkit
