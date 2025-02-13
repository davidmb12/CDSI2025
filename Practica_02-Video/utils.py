
import os
phaseDict = {
    1:"Complete",
    2:"Incomplete",
    0:"Null",
}
def renameFiles(directoryPath, name=''):
    try:
        files = sorted(os.listdir(directoryPath))  # Sort to ensure numbering is consistent
        method = 1
        
        for index, filename in enumerate(files, start=1):
            file_path = os.path.join(directoryPath, filename)
            if os.path.isfile(file_path):
                currentFileNumber = index % 3
                fileName, ext = os.path.splitext(filename)
                new_name = f"{name}_0{method}-{phaseDict[currentFileNumber]}{ext}"
                if currentFileNumber ==0:
                    method+=1

                new_path = os.path.join(directoryPath, new_name)
                os.rename(file_path, new_path)
                # print(f"Renamed: {filename} -> {new_name}")
    except Exception as e:
        print(f"Error: {e}")

renameFiles('./Videos/Jesus/',name='Jesus')
