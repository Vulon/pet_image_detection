cd /home/night
sudo apt update -y && sudo apt upgrade -y
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev -y
wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz
tar -xf Python-3.10.*.tgz
cd Python-3.10.*/
./configure --enable-optimizations
make -j 4
sudo make altinstall
sudo apt install python3-pip -y
sudo apt install git -y
cd ../

sudo pip3.10 install pipenv

git clone  https://github.com/Vulon/pet_image_segmentation.git
cd pet_image_segmentation
sudo pipenv install --dev --system
