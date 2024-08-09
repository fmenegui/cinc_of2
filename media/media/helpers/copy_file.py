import os

def copy_script_to_new_folder(script_path, target_folder):
    filename = os.path.basename(script_path)
    os.makedirs(target_folder, exist_ok=True)
    new_file_path = os.path.join(target_folder, filename)
    with open(script_path, 'r') as file:
        content = file.read()
    with open(new_file_path, 'w') as file:
        file.write(content)
    print(f"Copy of the script saved to: {new_file_path}")

if __name__ == '__main__':
    from media.helpers.copy_file import copy_script_to_new_folder
    script_path = __file__
    target_folder = 'path/to/your/new/folder'
    copy_script_to_new_folder(script_path, target_folder)