import os
import shutil


class IO:

    '''
    Provide basic read-write wrappers
    Additional utilities for setup/cleaning an experiment folder
    Misc:
        - generate a set of folders
        - clean a set of folders
    '''


    @staticmethod
    def setup_folders(keys, root: str):
        folders = {}
        for key in sorted(keys):
            base_folder = os.path.join(root, key)
            metric_folder = os.path.join(base_folder, 'metrics')
            checkpoint_folder = os.path.join(base_folder, 'checkpoints')
            if not os.path.exists(base_folder): os.mkdir(base_folder)
            if not os.path.exists(metric_folder): os.mkdir(metric_folder)
            if not os.path.exists(checkpoint_folder): os.mkdir(checkpoint_folder)
            folders[key] = (metric_folder, checkpoint_folder)
        return folders

    @staticmethod
    def is_dirty_folders(keys, root: str):
        for key in sorted(keys):
            base_folder = os.path.join(root, key)
            metric_folder = os.path.join(base_folder, 'metrics')
            checkpoint_folder = os.path.join(base_folder, 'checkpoints')
            if os.path.exists(metric_folder) and len(os.listdir(metric_folder)) > 0:
                return True
            if os.path.exists(checkpoint_folder) and len(os.listdir(checkpoint_folder)) > 0:
                return True
        return False

    @staticmethod
    def clean_folders(keys, root: str):
        for key in keys:
            base_folder = os.path.join(root, key)
            if os.path.exists(base_folder):
                shutil.rmtree(base_folder)