# What is this subdirectory?

The files here are for handling the "granular" model, which takes in separate CSV files for each drug category.

Patients in each dataset are not necessarily in the other datasets.

This model should be better for general drug prediction.

NOTE:  The Stanford files are WAY too big to be stored in GitHub, so you have to obtain them elsewhere.  
- You can get them here:  https://hivdb.stanford.edu/pages/geno-rx-datasets.html
- You should then remove the first six columns and the last two columns from each file.
- You should remove the first row from each file (we don't want to process the headers).