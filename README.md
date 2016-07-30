# LSTM-and-maxlayer-for-SNV-based-phenotype-prediction
Source code used in my MSc thesis "LSTM and convolutional neural networks for SNV-based phenotype prediction Long Short-Term Memory and convolutional neural networks for SNV-based phenotype prediction"


### Creation of artificial data ###

### Usage of real data (vcf) and conversion to hdf5 ###

### Reproducing maxlayer experiments ###

To start maxlayer experiments in a simple cross-validation, as done in my thesis for the artificial dataset, use the script maxlayer_start_cross-validation.py and specify the parameters to test, number of parallel processes, locations of input files, path for the results, and the device(s) to use for computation in the script. Then execute it with “python3.5 maxlayer_start_cross-validation.py” in the termial/console.
For nested cross-validation, as used with the bacterial and human datasets, use the script  maxlayer_start_nested_cross-validation.py analogously.
The trained networks and logfiles will be stored in the specified location.

For the evaluation of the results, first run the scripts analyze_logfiles.py or analyze_logfiles_nested_CV.py after specifying the respective paths in the script. This will create a summary-file of the results, as used for the preliminary evaluation in my thesis.

In case of the nested cross-validation, the file outer… can be used to calculate the scores on the test sets.

### Reproducing LSTM experiments ###

To start LSTM nested cross-validation experiments as in the thesis, use the script lstm_start_nested_cross-validation. Proceed as in section “Reproducing maxlayer experiments”.

