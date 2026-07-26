import glob
import os


def create_symlink(src, dst):
    '''
    Create a symlink from src to dst

    Parameters:
    ----------
    src : str
        source file
    dst : str
        destination file

    Returns:
    -------
    None
    '''
    parent = os.path.dirname(dst)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if not os.path.exists(dst):
        try:
            os.symlink(src, dst)
        except FileExistsError:
            os.unlink(dst)
            os.symlink(src, dst)
    else:
        print(f'{dst} already exists')


def remove_proc_files(files, dir):
    """
    Remove files that have already been processed in dir

    Parameters:
    ----------
    files : list
        list of all files in the data directory
    dir : str
        directory where some files have already been processed

    Returns:
    -------
    new_files : list
        list of files that have not been processed
    """
    proc_files = glob.glob(os.path.join(dir, 'raw', '*.fits'), recursive=True)
    proc_files = [os.path.realpath(i) for i in proc_files]
    new_files = list(set(files) - set(proc_files))

    return new_files
