# fastroc
fastroc is a module that is able to approximate the ROC AUC score over an axis of two numpy arrays much faster than current 
alternatives. It does this at the expense of portability as it uses C to calculate the ROC AUC.

## Usage

To calculate the ROC AUC score, 2 arrays are needed:
* `y_true` - Whether that event actually happened
* `y_score` - The model's score of that event happening. The lower, the more likely. `y_score` should be in the range 0 
   to 1.

The function also takes in 3 optional arguments:
* `axis` (default `-1`) - Which axis to compute the score over
* `integral_precision` (default `50`) - The number of samples to use for the integration
* `thread_count` (default `1`) - The number of threads to use. A value of 1 keeps the program single threaded.

The array returned contains the roc scores.

## Building

First run:
```bash
git clone https://github.com/brightlego/fastroc.git
cd fastroc
```
Then run:
### Linux:
To build and install with you default Python3 installation, run 
```bash
bash build.sh
```

### Other
To build and install with another Python3 installation or on MacOS/Windows, run
```bash
<python-installation> -m build
<python-installation> -m pip install "$(ls dist/*.whl | sort -V | tail -n 1)" --force-reinstall
```

#### Notes for Windows:
This code is not tested on Windows and is unlikely to be able to use multithreading.

