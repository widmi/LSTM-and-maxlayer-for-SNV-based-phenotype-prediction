# LSTM-and-maxlayer-for-SNV-based-phenotype-prediction
Source code used in my MSc thesis "LSTM and convolutional neural networks for SNV-based phenotype prediction Long Short-Term Memory and convolutional neural networks for SNV-based phenotype prediction"


### Creation of artificial data ###

The artificial datasets can be created with script generate_data.py as VCF files. VCF files can be converted to HDF5 files as described below.

### Input file creation (conversion from VCF to hdf5) ###
Standard VCF files can be converted to the required hdf5 type with the script vcf_to_hdf5_tools.py.


### Target file layout ###

The targets (true classes) of the samples have to be provided in tabulator-separated csv files (see example_target_file.csv for an example). The first column holds the sample names and other columns hold the classes. The first row holds the names of the classes. The relation from samples to classes can range from 0 to 1 and contain -1 for unknown sample labels.

For the nested cross-validation target files containing only the outer training or test sets are required, with the file extensions ‘.test_fold0’ and ‘.train_fold0’ for the test and training set respectively, where ‘0’ is replaced with the number of the fold (i.e 0-4 for a 5-fold outer cross-validation).
Standard cross-validation splits are done automatically and do not require extra files.


### Reproducing maxlayer experiments ###

To start maxlayer experiments in a simple cross-validation, as done in my thesis for the artificial dataset, use the script maxlayer_start_cross-validation.py and specify the parameters to test, number of parallel processes, locations of input files, path for the results, and the device(s) to use for computation in the script. Then execute it with “python3.5 maxlayer_start_cross-validation.py” in the termial/console.
For nested cross-validation, as used with the bacterial and human datasets, use the script  maxlayer_start_nested_cross-validation.py analogously.
The trained networks and logfiles will be stored in the specified location.

For the evaluation of the results, first run the scripts analyze_logfiles.py or analyze_logfiles_nested_CV.py after specifying the respective paths in the script. This will create a summary-file of the results, as used for the preliminary evaluation in my thesis.



### Reproducing LSTM experiments ###

To start LSTM nested cross-validation experiments as in the thesis, use the script lstm_start_nested_cross-validation. Proceed as in section “Reproducing maxlayer experiments”.

