# NobianHachathon


File Main.py or Main2.py Preforms the following functions 

Filter(Signal,dt): This function Takes the signal and makes it pass through a low band pass frequency filter in order to reduce noise in the signal 
ExcelPickler(): This function transforms the Excel sheet into a Pandas data-frame then creates a pickled document in order to easily load data 
                and reducing computational time to load the data 
Plot(data): This function plots the data of the valves according to the time stamp used 
find_last_indices(data): This function finds the last usable indices in the data sheet
drop_bad(data): this function filters out the data that is corrupted in the document, (valve A, valve C and valve B have some data that is "bad" or ""Unknown")
ROC(data,Start,End): was initially used to identify how many errors are present and also find the difference between valve A and C
fastfourier(Signal,dt): This function produces a fast fourier of the signal that is given to it
get_classes(data,Start,End): This function gives the 
