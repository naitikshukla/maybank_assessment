from data_prep import data_processing

filename = 'assessment.xlsx'
sheetname = 'Data'
df = data_processing(filename,sheetname)
df.head()
