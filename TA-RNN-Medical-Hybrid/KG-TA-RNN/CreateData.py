from MIMIC.pkl_files_preperation_old import pkl_files_creator
if pkl_files_creator() == -1:
    print('There is an error! Please run it again.')
else:
    print('Data is ready as pkl files.')
