import os

def get_absolute_path(current_file: str, file_name: str):
  current_dir = os.path.join(current_file, file_name)
  return os.path.abspath(current_dir)
