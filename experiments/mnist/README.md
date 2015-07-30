MNIST
=====

This is an example of how the experiment directory should look like. For your convinience this directory includes the data and has all the subdirectories for demonstation purposes.  

Running
-------

To run this example execute the following (assuming CWSM_ROOT in the root of the repository)
```
cd CWSM_ROOT
export PYTHONPATH=CWSM_ROOT:$PYTHONPATH
python run.py --experiment experiments/mnist --optimize accuracy
```

Output
------

The output will be something like this
```
$ python run.py --experiment experiments/mnist --optimize accuracy
WARNING: image mean file was not found: experiments/mnist/data/mean_train.binaryproto. Ignore this message if you know that you do not need it.
about to fork child process, waiting until server is ready for connections.
forked process: 24465
child process started successfully, parent exiting
Removing previous results and temporary files ...
Cleaning up experiment cafferun in database at localhost
rm: cannot remove `experiments/mnist/caffeout/*': No such file or directory
Using database at localhost.
Getting suggestion...

Suggestion:     NAME          TYPE       VALUE
                ----          ----       -----
                num_output_1  int        1
                weight_decay  float      0.000100
                num_output_3  int        3
                num_output_2  int        3
                momentum_1    float      0.800000
                base_lr_1     int        1
                kernel_size_  int        3
                kernel_size_  int        3
Submitted job 1 with local scheduler (process id: 24503).
Status: 1 pending, 0 complete.
```
If you see similar output then system is working. If new suggestions will appear too quickly it means that something went wrong. Check `experiments/mnist/spearmint/output/*.out` files to see STDOUT foe each partiuclar run and look inside `experiments/mnist/caffeout/DATETIME_log.txt` for Caffe output.

Results
-------
The network with default parameters for the MNIST dataset provided in [Caffe examples](https://github.com/BVLC/caffe/tree/master/examples/mnist) achieves **~0.99**.  
With the parameters
```
TODO: re-run and paste parameters
```
proposed by CWSM it is possible to squeeze **~0.995**
