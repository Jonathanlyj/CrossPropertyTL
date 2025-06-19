pip install -r torch_requirements.txt
cd data/
wget https://figshare.com/ndownloader/articles/27569631/versions/2 -O tl_dataset.zip
unzip tl_dataset.zip -d tl_dataset
cd ../elemnet/
wget https://figshare.com/ndownloader/articles/29367467?private_link=1be8bdcbd8db1ff52d5c -O models.zip
unzip models.zip

