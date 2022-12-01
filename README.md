# Muscle Fatigue Detection

This projects contains the necessary tools to detect muscle fatigue from sEMG signals.

## Description
In ``` experiment.ipynb``` you will see examples of analysing signals from dataset located under 'subject_data'.
the analysis is done in ``` analyze.py ```. Where, the analysis processing is in ``` signal_processing.py``` file.

All figures ploting are grouped in ``` plotting.py```

## Getting Started

### Dependencies

* matplotlib.pyplot
* import pandas
* import numpy 
* import scipy 
* fftpack from scipy 
* detect from biosignalsnotebooks

### Installing
* The dataset should be placed in folder with name '<Subject_Name>_data'

### Executing program

* Create folder for the dataset with name '<Subject_Name>_data'
* Use the functions in ``` analyze.py``` to run the analysis.
e.g. 
```
import analyze as an

an.analyze_csv(subject="Lubos", file="right_arm" , channels=[1,2,3,14,15,16])

```

## Help

Please reach out in case help is needed.
See Contact infor below.


## Authors
Ismail Farah
Email: ismai1-farah@hotmail.com

## Version History

* 0.2
    * Clean up & removing unneeded files
    * See [commit change]()
* 0.1
    * Initial Release

## Acknowledgments
Inspiration, code snippets, etc.
* [biosignalsnotebooks
](https://github.com/pluxbiosignals/biosignalsnotebooks)
