## About DeePhy

__`DeePhy`__ is a deep learning framework for inferring triplet tree topology from three unaligned nucleotide sequences.  


## Required packages

The DefIn is developed on python 3.6 and PyTorch 1.9. This tool is tested on Linux based system. The prerequisites are as follows,

- Python 3.6
- Numpy 1.19
- scikit-learn 0.24.2 
- scipy 1.5.4
- BioPython 1.79
- Json 0.1.1
- PyTorch 1.9
- CudaToolkit 10.2.89
- torchvision 0.10.0

As an alternative, a conda environment, `dphy.yml`, is also provided in this package. 


## DeePhy execution

# Step-1: Derive GFP

DeePhy takes Genomic Footprint (GFP) of triplet data. Hence, GFP of nucleotide sequence is required to derive. 
The tool for deriving GFP a nucleotide sequence is provided in the following link,
http://www.facweb.iitkgp.ac.in/~jay/GRAFree2/GRAFree2.html

# Step-2 (Optional): Create ground truth

Create the ground truth of the triplet tree(s). Three OTUs are one-hot encoded. The siblings and outgroup are denoted by 0 and 1, respectively.

# Step-3: Prepare test data

This step is used to partition entire dataset into training, validation, and test data. For only testing with the trained model, set training and 
validation as blank list.

# Step-4: Execute DeePhy

Based on the training, validation, and testing partition DeePhy executes prediction of triplet topology.


## Execution of DefIn

`python main.py --dataset [location of dataset] --subdir [name of subdirectory]`

Options:

`--dataset`	dataset path
`--subdir`	subdirectory of dataset, e.g. GFP
`--workers`	number of data loading workers (default=16)
`--batch_size`	batch size (default=512)
`--cuda`	the program uses CUDA
`--outf`	output folder
`--save`	path to saved model
