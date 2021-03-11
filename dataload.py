from google_drive_downloader import GoogleDriveDownloader as gdd
import json 
import os 

def data_download(dest_path: str, category: str):
    """
    Data doanload using google_drive_downloader
    https://github.com/ndrplz/google-drive-downloader

    Argument
    --------
    dest_path: str
        directory where to save file. ex) ../data
    category: str
        category for downloading data
    """
    try:
        print('Start Download')
        gdd.download_file_from_google_drive(file_id='1EKYU6nL0vRs-7sV7g0E_4OJVRlY7LLYC',
                                            dest_path=os.path.join(dest_path,f'{category}.json'),
                                            unzip=False,
                                            overwrite=True)
        print('End Download')
    except Exception as e:
        print(e)

def load_data(path: str, category: str='order', download: bool=False):
    """
    Load data from path

    Arguments
    ---------
    path: str
        saved data directory. ex) ../data
    category: str
        category for loading data
    download: bool
        if True downloads data

    Return
    ------
    data: json
    """
    if download:
        data_download(path, category)

    data = json.load(open(os.path.join(path,f'{category}.json'),'r'))

    return data



    