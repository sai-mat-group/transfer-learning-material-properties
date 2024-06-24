# Pre-Train Mulitprop model

The multi-property model uses a different architecture than the single property models. It is based on ALIGNN, but has multiple MLPs for prediction. The code for these models is in the the `../alignn_files` directory - these models are loaded in the trianing and testing scripts in this directory. Please either add `../alingnn_files` to your path or copy those files here.

In the files you must also specify which data from the compiled dataset you want to drop, this is the `drop_indices` parameter. The indices for the properties are:
0 - GVRH
1 - PHONONS
2 - PIEZO
3 - DIELECTRIC
4 - EXP_BAND_GAP
5 - FORM_ENERGY
6 - BAND_GAP

The best trained models for all of these are already available in `../../SEPARATE-HEAD-PT-MODELS`
