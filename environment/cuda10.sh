大概的步骤是
1.sudo dpkg -i nvidia-driver-local-repo-ubuntu1804-450.80.02_1.0-1_amd64.deb
2sudo apt install -y gcc g++ build-essential linux-headers-$(uname -r)
3.sudo dpkg -i /data/cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
4.sudo apt-get update && sudo apt-get install cuda-10.0 
5.安装cudnn
6.重启系统
