import pandas as pd

def csvLoader(filename, defaultPath = 'data/'):
    df = pd.read_csv(defaultPath+filename)
    return df

def loadAllCSVfromFolder(folderPath = 'data/'):
    import os
    files = os.listdir(folderPath)
    csvFiles = []
    for file in files:
        if file.endswith(".csv"):
            csvFiles.append(file)
    csv_files_list = []
    for i in range(len(csvFiles)):
        df = csvLoader(csvFiles[i], folderPath)
        csv_files_list.append(
            {
                "df": df,
                "filename": csvFiles[i].split(".")[0],
            }
        )
    return csv_files_list