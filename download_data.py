import os
import pathlib
import gdown
import shutil

# NOTE: For development, all data is stored in /home/shareFolder/DA_datasets
# https://www.geeksforgeeks.org/how-to-create-a-shared-folder-between-two-local-user-in-linux/

data_root = "data"
pathlib.Path(data_root).mkdir(parents=True, exist_ok=True)
os.chdir(data_root)

# Office-Home
# Original URL: http://hemanthdv.org/OfficeHome-Dataset/
filename = "OfficeHome.zip"
gdown.download("https://drive.google.com/uc?id=1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC&confirm=t", filename)
shutil.unpack_archive(filename)
os.remove(filename)
shutil.move('OfficeHomeDataset_10072016', 'office_home')
shutil.move('office_home/Real World', 'office_home/Real')

# DomainNet
# Original URL: http://ai.bu.edu/M3SDA/
data_folder = 'domain_net'
pathlib.Path(data_folder).mkdir(parents=True, exist_ok=True)
urls = [
    "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip",
    "http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip",
    "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip",
    "http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip",
    "http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip",
    "http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip"
]
for url in urls:
    filename = os.path.join(data_folder, url.split("/")[-1])
    gdown.download(url, filename)
    shutil.unpack_archive(filename, data_folder)
    os.remove(filename)