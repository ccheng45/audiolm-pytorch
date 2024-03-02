import urllib.request
import os
import tarfile


def download_progress(block_num, block_size, total_size):
    """
    Callback function to report download progress.

    Parameters:
    - block_num: The current block number.
    - block_size: The size of each block.
    - total_size: The total size of the file.
    """
    downloaded = block_num * block_size  # Bytes downloaded so far
    progress = 100.0 * downloaded / total_size  # Current progress percentage
    progress = min(100, progress)  # Ensure progress does not go over 100%
    print(f"\rDownload progress: {progress:.2f}%", end="")


# Ensure the data directory exists
data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Path where the downloaded file will be saved, relative to the current directory
save_path = os.path.join(data_dir, 'dev-clean.tar.gz')
libr = os.path.join(data_dir, 'LibriSpeech')


url = "https://us.openslr.org/resources/12/dev-clean.tar.gz"

print("Downloading the file...")
if not os.path.isfile(save_path):
    urllib.request.urlretrieve(url, save_path, reporthook=download_progress)

print("Extracting the file...")
if not os.path.isdir(libr):
    with tarfile.open(save_path, "r:gz") as tar:
        tar.extractall(path=libr)
    print("Extraction completed.")

# Hubert
hubert_ckpt = "hubert/hubert_base_ls960.pt"
hubert_quantizer = f"hubert/hubert_base_ls960_L9_km500.bin"  # listed in row "HuBERT Base (~95M params)", column Quantizer

if not os.path.isdir("hubert"):
    os.makedirs("hubert")
print("downloading hubert ckpt")
if not os.path.isfile(hubert_ckpt):
    
    hubert_ckpt_download = f"https://dl.fbaipublicfiles.com/{hubert_ckpt}"
    urllib.request.urlretrieve(
        hubert_ckpt_download, f"./{hubert_ckpt}", reporthook=download_progress
    )
print("downloading hubert kmean")
if not os.path.isfile(hubert_quantizer):
    hubert_quantizer_download = f"https://dl.fbaipublicfiles.com/{hubert_quantizer}"
    urllib.request.urlretrieve(
        hubert_quantizer_download, f"./{hubert_quantizer}", reporthook=download_progress
    )
