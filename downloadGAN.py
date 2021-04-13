
import math
import os
import hashlib
from urllib.request import urlretrieve
import zipfile
import gzip
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm


class DLProgress(tqdm):
    """
    Handle Progress Bar while Downloading
    """
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        """
        A hook function that will be called once on establishment of the network connection and
        once after each block read thereafter.
        :param block_num: A count of blocks transferred so far
        :param block_size: Block size in bytes
        :param total_size: The total size of the file. This may be -1 on older FTP servers which do not return
                            a file size in response to a retrieval request.
        """
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def _unzip(save_path, _, database_name, data_path):
    """
    Unzip wrapper with the same interface as _ungzip
    :param save_path: The path of the gzip files
    :param database_name: Name of database
    :param data_path: Path to extract to
    :param _: HACK - Used to have to same interface as _ungzip
    """
    print('Extracting {}...'.format(database_name))
    with zipfile.ZipFile(save_path) as zf:
        zf.extractall(data_path)


DATASET_CELEBA_NAME = 'celeba'
data_path = './celeba'
url = 'https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip'
hash_code = '00d2c5bc6d35e252742224ab0c1e8fcb'
extract_path = os.path.join(data_path, 'img_align_celeba')
save_path = os.path.join(data_path, 'img_align_celeba.zip')
extract_fn = _unzip

if os.path.exists(extract_path):
    print('Found {} Data'.format(DATASET_CELEBA_NAME))


if not os.path.exists(data_path):
    os.makedirs(data_path)

if not os.path.exists(save_path):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Downloading {}'.format(DATASET_CELEBA_NAME)) as pbar:
        urlretrieve(
            url,
            save_path,
            pbar.hook)

assert hashlib.md5(open(save_path, 'rb').read()).hexdigest() == hash_code, \
    '{} file is corrupted.  Remove the file and try again.'.format(save_path)

os.makedirs(extract_path)
try:
    extract_fn(save_path, extract_path, DATASET_CELEBA_NAME, data_path)
except Exception as err:
    # Remove extraction folder if there is an error
    shutil.rmtree(extract_path)
    raise err

# Remove compressed data
# os.remove(save_path)
