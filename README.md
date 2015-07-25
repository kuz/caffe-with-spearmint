Caffe with Spearmint (CWSM)
===========================
Automate [Caffe](https://github.com/BVLC/caffe) parameter search via [Spearmint](https://github.com/HIPS/Spearmint) Bayesian optimisation. This tool is meant for people who use caffe and are tired of trying to fit parameters manually. For those familiar with Caffe: the only (almost) thing you will to differently is instead of
```
weight_decay: 0.05
```
you now can write
```
weight_decay: OPTIMIZE{"type": "FLOAT", "min": 0, "max": 0.2}
```
and CWSM will take it form there. The `OPTIMIZE` keyword can be used both in `solver.prototxt` and `trainval.prototxt`. There are few special tricks which may seem odd at first, but, as you will see below, they are justified. Refer to "Optimization Parameters" section below for more detailed documentation.


General Setup
-------------

#### STEP 1: Install Caffe
Follow the instructions in the [Caffe](https://github.com/BVLC/caffe) repository. Once installed set the `CAFFE_ROOT` variable to point to the Caffe repository root so that `CAFFE_ROOT/build/tools/caffe` would be the Caffe binary.

#### STEP 2: Install Spearmint
Follow the instruction in the [Spearmint](https://github.com/HIPS/Spearmint) repository. Once install set the `SPEARMINT_ROOT` variable to point to the Spearmint repository root so that `SPEARMINT_ROOT/spearmint/main.py` is the Spearmint main script.

#### STEP 3: Prepare experiment directory
The only parameter to CWSM's `run.py` script needs is the location of the experiment directory. This directory must have quite specific structure: It his looks scary just create the root experiment folder and run `python run.py experiments/myexperiment` and the script will tell you what is missing.
```
myexperiment
  caffeout                  # [created automatically] will be filled with Caffe output
  data                      # holds your data files
    mean_train.binaryproto  # image means computed with Caffe "compute_image_mean"
    train_lmdb              # LMDB directory for the training set
      data.mdb
      lock.mdb
    val_lmdb                # LMDB directory for the validation set
      data.mdb
      lock.mdb
  model                     # here you describe the model
      solver.prototxt       # yes, it has to be named exactly like that
      trainval.prototxt     # this one too
  mongodb                   # [created automatically] Spearmint holds its stuff here
  spearmint                 # [created automatically] Spearmint config and optimization function definition
    output                  # [created automatically] Spearming output files
  tmp                       # [created automatically] experiment run specific data
```

#### STEP 4: Describe your Caffe model


#### STEP 5: Run the optimization

Optimization Parameters
-----------------------

#### INT

#### FLOAT

#### ENUM
