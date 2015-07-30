Caffe with Spearmint (CWSM)
===========================
Automatic [Caffe](https://github.com/BVLC/caffe) parameter search via [Spearmint](https://github.com/HIPS/Spearmint) Bayesian optimisation. For those familiar with Caffe: now instead of specifying 
```
weight_decay: 0.05
```
in your `solver.prototxt` you can write
```
weight_decay: OPTIMIZE{"type": "FLOAT", "min": 0, "max": 0.2}
```
and CWSM will use Spearmint to find the best `weight_decay` for you.  

For those not familiar with Caffe: you might have heard that choosing right parameters for a deep neural network is a painful process. This tool employs the power of Bayesian optimization to make this search automatic for the models described in the Caffe framework.

The `OPTIMIZE` keyword can be used both in `solver.prototxt` and `trainval.prototxt` for as many parameters simultaneuosly as you want (be careful: too many parameters might take too long).  
  
There are few special tricks which may seem odd at first, but, as you will see below, they are justified. Take a look at [`experiments/mnist`](experiments/mnist) and read the "General Setup" section for a quick start. Refer to "Optimization Parameters" section below for more detailed documentation.


General Setup
-------------

#### STEP 1: Install Caffe
Follow the instructions in the [Caffe](https://github.com/BVLC/caffe) repository. Once installed set the `CAFFE_ROOT` variable in the `run.py` script to point to the Caffe repository root so that `CAFFE_ROOT/build/tools/caffe` would be the Caffe binary.

#### STEP 2: Install Spearmint
Follow the instruction in the [Spearmint](https://github.com/HIPS/Spearmint) repository. Once install set the `SPEARMINT_ROOT` variable in the `run.py` scipt to point to the Spearmint repository root so that `SPEARMINT_ROOT/spearmint/main.py` is the Spearmint main script.

#### STEP 3: Prepare experiment directory
The main parameter CWSM's `run.py` script needs is the location of the experiment directory. Create your `experiments/myexperiment` and you are ready to go to the next step, `run.py` will take care of creating the structure and will tell you what to do next.  
For the refernce, the final structure will look like this:
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
Under `experiments/myexperiment/model` CWSM will expect to find two (in special cases three) files:  
**`solver.prototxt`**  
Take a look at [`experiments/mnist/model/solver.prototxt`](experiments/mnist/model/solver.prototxt), pay attention to the `max_iter` parameter -- it defines how long each run will take.  
**`trainval.prototxt`**  
This file is used during training, if you are going to optimize accuracy you should have a layer named "accuracy", for example
```
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "fc2"
  bottom: "label"
  top: "accuracy"
}
```
If you are optizing loss then you should have a layer named "loss", something like
```
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "fc2"
  bottom: "label"
  top: "loss"
}
```
**`val.prototxt`**  
This file is needed if you use an evaluation parameters which is not built-in in Caffe. Currently there is only one such implemented: squared kappa. To use that you will need to create `val.prototxt` which mimicks `trainval.prototxt`. The difference is that instread of `loss`/`accuracy` layers `val.prototxt` should have a layer be named `prob`. This layes is used to extract predictions used to calculate kappa.
```
layer {
  name: "prob"
  type: "Softmax"
  bottom: "fc2"
  top: "prob"
}
```

#### STEP 5: Run the optimization
Make sure you have Caffe in your `$PYTHONPATH`.  
**The `run.py` script has to be started from the repository root:**  
Assuming `CWSM_ROOT` is the location of this repository, do the following:  
```
cd CWSM_ROOT
export PYTHONPATH=CWSM_ROOT:$PYTHONPATH
```
Now you are ready to run!
```
python run.py --experiment experiments/mnist --optimize accuracy
```

#### STEP 6: Enjoy!
Leave the processes running for a night and in the morning have a look at the latest
```
Minimum expected objective value under model is -0.99590 (+/- 0.08043), at location:
<...>
```
message in `STDOUT`. There is a chance that it will provide you with the best parameter configuation you've seen so far.

Optimization Parameters
-----------------------
Spearmint support three types of variables: `INT`, `FLOAT` and `ENUM`. Here is how you use them in CWSM.

#### INT
Default way is `OPTIMIZE{"type": "INT", "min": 2, "max": 5}`, for example if you want to try different strides for a convolutional layer.
```
layer {
  name: "conv1"
  type: "Convolution"
  <...>
  convolution_param {
    <...>
    stride: OPTIMIZE{"type": "INT", "min": 2, "max": 5}
    <...>
  }
}
```
There are useful additional tricks for `INT`, see them below.

#### FLOAT
Written as `OPTIMIZE{"type": "FLOAT", "min": 0, "max": 0.2}`. For example
```
weight_decay: OPTIMIZE{"type": "FLOAT", "min": 0, "max": 0.2}
```
in your `solver.prototxt`.  
Note that finding the right `FLOAT` parameter will take longer than finding `INT`.

#### ENUM
Written as `OPTIMIZE{"type": "ENUM", "options": ["2", "17", "100"]}`, for example
```
lr_policy: OPTIMIZE{"type": "ENUM", "options": ["fixed", "inv", "step"]}
```
in your `solver.prototxt`.  
Note that `ENUM` loses the relative information between the options: 0.5 is not less than 0.6 anymore, they are completely unrelated values when used with with `ENUM`.

#### Important Tricks
There are many cases where you would like to keep relative information between the values and avoid using `ENUM`. For example if you want to try {100, 200, 300} as sizes for a fully connected layer. Or try learning rates {0.0001, 0.001, 0.01} but avoid using `FLOAT` because you do not want to waste time on 0.0101 etc. For these cases CWSM has a special `transform` parameter which can be used with `INT`. Here are examples what is can do with intuition why this might be a good idea.

##### X<N>
Written as `OPTIMIZE{"type": "INT", "transform": "X10", "min": 1, "max": 5}`. Here `"transform":"X10"` acts as a multiplier where instead of 10 you can use any number stands for any number ("X10", "X100", "X22"). The effect of this particular line is that numbers {10, 20, 30, 40, 50} will be tried as a parameters.  
Another example would be 
```
layer {
  name: "fc1"
  type: "InnerProduct"
  <...>
  inner_product_param {
    num_output: OPTIMIZE{"type": "INT", "transform": "X100", "min": 1, "max": 3}
    <..>
  }
}
```
which will try {100, 200, 300} as the size of the fully connected layer.  
Reasoable thing to ask it why not to use `ENUM` for this as `num_output: OPTIMIZE{"type": "ENUM", "options": ["100", "200", "300"]}`? Because the Bayesian optimizer will not be able to use the fact that 100 < 200 anymore.

##### NEGEXP<N>
Where <N> is the base, the number which goes to Spearmint is negative of the exponent (example: value 3 with `NEGEXP10` means 10^-3 and correpsonds to 0.001). Can be used for learning rate in `solver.prototxt`:
```
base_lr: OPTIMIZE{"type": "INT", "transform": "NEGEXP10", "min": 2, "max": 5}
```
This will try {0.01, 0.001, 0.0001, 0.0001} keeping in mind their order and without the need to try all intermediate values as `FLOAT` would do.

##### LOG<N>
Number which goes to Spearmint corresponds to log with base <N> of an actual number (example: value 2 of LOG2 corresponds to 4). Would be handy if you want to try 2,4,8,16,... for some parameter.

TODO
----
* Extract summary from MongoDB in some readable form
