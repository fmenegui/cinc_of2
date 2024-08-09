''' Tranform #Dx: NORM to #Dx: Normal
       and #Dx: MI, ..., to #Dx: Abnormal
'''
import sys
import os

def replace_line_in_file(file_path):
    if not file_path.endswith('.hea'):
        return  # Skip files that do not have the .hea extension
    with open(file_path, 'r') as file:
        lines = file.readlines()
    with open(file_path, 'w') as file:
        for line in lines:
            if line.strip().endswith("#Dx: NORM"):
                file.write(line.replace("NORM", "Normal"))
            elif "#Dx: " in line and not line.strip().endswith("NORM") and not line.strip().endswith("#Dx:"):
                file.write(line.replace(line[line.index("#Dx: ") + 5:].strip(), "Abnormal"))
            elif line.strip().endswith("#Dx:"):  # This checks if the line ends with '#Dx:' indicating it is empty
                file.write(line.rstrip() + " Abnormal\n")  # Appends 'Unknown' to the '#Dx:' line
            else:
                file.write(line)

def correct_ptbxl_binary(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        for filename in files:
            file_path = os.path.join(subdir, filename)
            print(file_path)
            replace_line_in_file(file_path)
            

if __name__ == '__main__':
    print(sys.argv)
    correct_ptbxl_binary(sys.argv[1])
