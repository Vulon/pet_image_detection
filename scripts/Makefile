cd /home/night
sudo apt update -y && sudo apt upgrade -y
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev liblzma-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev -y
wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz
tar -xf Python-3.10.*.tgz
cd Python-3.10.*/
./configure --enable-optimizations
make -j 4
sudo make altinstall
sudo apt install python3-pip -y
sudo apt install git -y
cd ../



sudo apt install ca-certificates curl gnupg lsb-release
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt install apt-transport-https ca-certificates curl gnupg2 software-properties-common -y
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/debian $(lsb_release -cs) stable"
sudo apt update
apt-cache policy docker-ce
sudo apt install docker-ce -y

git clone  https://github.com/Vulon/pet_image_detection.git
cd pet_image_detection

python3.10 -m venv venv
venv/Scripts/activate
sudo pip3.10 install pipenv
sudo pipenv install --dev

sudo usermod -aG docker $USER
newgrp docker
