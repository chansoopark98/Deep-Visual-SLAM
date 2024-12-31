# Deep-Visual-SLAM
Deep based Visual SLAM(VO/VIO)

# NVIDIA Driver Remove 

sudo apt-get purge *nvidia*
sudo apt-get autoremove
sudo apt-get autoclean

# CUDA Remove

sudo rm -rf /usr/local/cuda*
sudo apt-get --purge remove '*cud*'
sudo apt-get autoremove --purge '*cud*'

# 잔여 패키지 제거 확인
sudo dpkg -l | grep nvidia
sudo dpkg -l | grep cuda
# 패키지 삭제
sudo apt-get remove --purge <패키지명>



https://dimlrgbd.github.io/rawdata