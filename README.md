# PIONEER pipeline installation and setup
PIONEER is a deep learning-based framework that predicts protein interfaces using a maximal set of sequence- and structure-based features. To access the PIONEER web interface, please go to our [web version](https://pioneer.yulab.org/).

## Step 1: Creating a conda environment
After cloning the repository, there is a quick way to install an environment for PIONEER. Please just run the following command lines (If the environment is installed without any errors, you can just go to Step 2):
```
conda create -n pioneer python=3.8.1
conda activate pioneer
conda install numpy=1.19.2
conda install pandas=1.2.4
conda install -c numba numba=0.53.1
conda install -c bioconda cd-hit=4.8.1
conda install -c pytorch pytorch=1.4.0 cudatoolkit=10.1
pip install torchvision==0.5.0
conda install -c ostrokach-forge torch-scatter=1.4.0 torch-sparse=0.4.3 torch-cluster=1.4.5 torch-spline-conv=1.1.1
pip install torch-geometric==1.3.2
```
Now you should have an environment where PIONEER can be run smoothly.

## Step 2: Required third-party software
The following third-party software should be required:\
[makeblastdb](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/)\
[blastp](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/)\
[psiblast](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/)\
[clustalo1.2.4](http://www.clustal.org/omega/)\
[naccess2.1.1](http://www.bioinf.manchester.ac.uk/naccess/)\
[zdock3.0.2](https://zdock.umassmed.edu/software/download/)\
[RaptorX](https://github.com/realbigws/RaptorX_Property_Fast)

## Step 3: Setting up the correct paths in config.py
Please open pioneer/config.py and change every path. You need to have all the binaries and third-party software installed in application directories accordingly. For the data cache, you can create an empty folder for each of the cache folders. If you're using empty cache folders, please refer to the next step for required directories and files within these empty cache folders.

The following files in config.py should be downloaded:\
[uniprot_seq_dict.pkl](https://pioneer.yulab.org/downloads)\
[uniprot_all.fasta](https://pioneer.yulab.org/downloads)\
[SASA_perpdb_alltax.txt](https://pioneer.yulab.org/downloads)\
[SASA_AF_perpdb_alltax.txt](https://pioneer.yulab.org/downloads)\
[pdbresiduemapping.txt](https://pioneer.yulab.org/downloads)\
[AF_predictions.pkl](https://pioneer.yulab.org/downloads)

## Step 4: Required directories and files within cache folders
Even if you work with an empty data cache, there are directories and files that are required for some of the cache folders. In the ModBase cache (MODBASE_CACHE in pioneer/config.py), you need to have the following subdirectories:
```
 -- modbase_cache
 | -- models
   | -- hash
   | -- header
   | -- uniprot
 | -- parsed_files
```
In the ZDOCK cache (ZDOCK_CACHE in pioneer/config.py), you need to have the following directories:
```
 -- zdock_cache
 | -- pdb_docked_models
 | -- mb_docked_models
 | -- mixed_docked_models
```

## Step 5: Running PIONEER
Please enter the root folder of the github repository, and execute the following command as an example:
```
python run_pioneer.py test_interactions.txt result_folder
```
The results can be found in the result_folder. To avoid repeatedly calculating the features for the same inputs in your next running, you can copy the calculated features from each feature folder in the result_folder/non_cached_files over to the corresponding cache directory indicated in pioneer/config.py.
