Usage:
spk-correlation.py correlation_config.json project_simulations.txt output_fn.pkl

- The entries of "correlation_config.json" are explained below
- project_simulations.txt is a text file where each line specifies the path to one simulation in the campaign. If the campaign was generated with "bbp-workflow", this file is at the campaign root
- output_fn.pkl is where the results will be written (pandas.DataFrame into pickle)

Configuration file contents:

 - base_target (Optional, str): Limits the analysis to that target (specifically: the overlap of that target and the simulation CircuitTarget). Default: "Mosaic"
 - t_start (Optional, float): Spiking activity before that point is ignored (Default: 0.0)
 - t_end (Optional, float): Spiking activity after that point is ignored (Default: time of last spike)
 - binsize (float): Spiking activity will be binned in bins of that size (in ms).
 - n_controls (Optional, int): How many random controls to generate by shuffling spiking times. (Default: 0)
 - correlation (dict): Specifies the type of correlogram:
    - t_win (Optional, list of floats): Correlation is calculated for delta t offsets between those two values (in ms). Default: [-250.0, 250.0]
    - type (str): One of: "sta", "convolution". If "convolution", the correlation is calculated as the convolution of the normalized spiking histograms of the populations
                                                If "sta", it is calculated as the spike triggered average of the normalized spiking histogram of one population, triggered by spikes of the other population
    - subsample (Optional, float/int): Only if type == "sta":
                             Spike triggered average is calculated using the spikes of only a subset of the other population. Speeds calculation up.
                             If a float is specified, it is the fraction of neurons in the other population to consider. If an int is specified it is the number of neurons to consider.
                             (Default: 1.0, i.e. no subsampling)
    - return_type (Optional, str): Only if type == "sta":
                         One of "individuals", "mean". If "mean", then the mean of the spike triggered average over neurons in the other population is returned.
                         If "individuals", then spike triggered averages of individual neurons are returned. Warning: result can get very large!
                         (Default: "individuals")
 - neuron_classes (dict): Defines the classes of neurons to consider.
                          Each key of the dict labels a neuron class. The values is another dict, where each key/value pair defines a neuron property and a list of valid values.
                          Example: {"L2_EXC": {"layer": [2], "synapse_class": ["EXC"]}}
 - condition_filter (dict, Optional): Only analyzes simulations with the specified simulation conditions.



----

spk-correlation-workflow.py is a functionally identical version that takes inputs as specified in https://bbpgitlab.epfl.ch/conn/simulation/sscx-analysis/-/issues/3

