# Distributed Inference Estimation and Optimization
This is the ___ project. This code and data have been used for the works 

- DIME: Distributed Inference Model Estimation forMinimizing Profiled Latency, *Robert Viramontes and Azadeh Davoodi*. [Published in SmartComp 2024](https://doi.org/10.1109/SMARTCOMP61445.2024.00081)

## Related Publications

- DIME at SmartComp 2024 in Osaka, JP
- PhD Forum at SmartComp 2024 in Osaka, JP
- PhD Forum at Design and Automation Conference 2024 in San Francisco, CA

## Requirements

Python 3.6+

[Git LFS](https://git-lfs.com/) (particularly, `alexnet_samples`, `vgg11_samples`, and `vit_samples` are tracked with LFS)

[gurobipy](https://www.gurobi.com/documentation/current/refman/py_python_api_overview.html), with appropriate Gurobi license. Tested with gurobipy 10.0.2.

PyTorch

### Recommendations
Flake8 for linting

Black for code formatting

## How to Use
There are 3 high-level steps in profiling and optimizing a network:

1. Generate intermediate feature maps for the network of interest, to be used during the profiling process.
2. Profile the network for bundle execution timing, and optionally collect power logs.
3. Generate the ILP and run the optimization process.

Each of these steps roughly corresponds to a particular code file. We will discuss these steps in detail below, using AlexNet network as the reference. 

### Generate intermediate feature maps

For profiling, we use a small subset (20 images) from the ImageNette dataset to profile over typical input data. ImageNette contains 10 classes, so 2 images were chosen at random from each class. The selected images can be found in [imagenette_samples](imagenette_samples).

We profile bundles, which may not start at layer 1. To ensure we have realistic values for all profiles bundled, we must collect the inputs for all layers in the DNN to be profiled so a bundle starting at any layer may be profiled. This may be done by running
```
python3 generate_alexnet_samples.py
```
This will collect the trained weights for AlexNet and run each of the samples in `imagenette_samples` through AlexNet, saving the output of each layer. The outputs are saved in [alexnet_samples](alexnet_samples/) and organized into folders named by the layer they are input for. Note that the file names match the source file names from `imagenette_samples`.  

> [!NOTE]
> We've already included the intermediate feature maps for AlexNet, VGG11, and ViT so you do not need to generate these on your own. 

### Profile network
Profiling must be done on each device you intend to include in the solution space. 

We support profiling on both CPU and GPU. Currently, these are customized to specific devices because of the unique set of frequenceis available on each device for voltage-frequency-dependent profiling (see FreDDI). 

The profiling scripts require the device to be specified with the device arg `-d (--device)`. Specifying `nano` or `potato` will trigger customized profiling based on the device characteristics, any other argument will lead to a generic profiling that will profile on GPU if available, otherwise CPU. 

Profiling also accepts optional arguments to specify the number of iterations `-i (--iterations)` and runs `-r (--runs)`. If collecting power information from the SmartPower 3, then the host server where the application is running should be specified with IP address `--log-server-addr` and port `--log-server-port` so the DUT can trigger logging on the host. See [powerlogging readme](smartpower3/README.md) for more info.

For example, to profile AlexNet on a Jetson Nano and connected to a host device with IP 192.168.0.1 with power logging server running at port 8001

```
python3 segmented_alexnet_profile.py -d nano --log-server-addr 192.168.0.1 --log-server-port 8001
```

This will save the timing information local to the DUT in a CSV file. Power information will be saved to a CSV on the host device. 

> [!IMPORTANT]
> If you are utilizing power logging and want to determine the energy of the bundles, then the timing and power profiles must be combined to generate joint timing-energy profiles. See more in [powerlogging readme](smartpower3/README.md).


### Run optimization
Once profiling is complete for all devices to consider, you can run the optimization process to find an optimal schedule. 

To generate an optimized schedule for the AlexNet network, first we update [alexnet_ilp_solver.py](alexnet_ilp_solver.py) to correctly point to the profiles. This is done at the top of the `run*_optimization`, where the edge device profile name is specified first and cloud second. 

The optimization can be run with
```
python3 alexnet_ilp_solver.py
```

This will generate several outputs:
- Will log solutions for minimum latency, and when using energy profiles, minimum energy. Will also print schedules compared to fast-only and slow-only, when doing DVFS comparisons.
- If doing `run_energy_optimization`, will save the results of contraint sweeps to `alexnet_DUT_min_OPTIMIZATIONGOAL_df.csv`. 

The output will look something like:
```
('conv1_conv2_conv3_conv4_conv5', 691200000, 'nano'): 1.0
('fc1_fc2_fc3', -1, 'wiscad'): 1.0
    latency: 0.06728694681524951 comp latency: 0.016872313541546464 comm_latency:  0.05041463327370304
    energy: 91600.34829106425 comp energy: 89580.15625 comm energy: 2020.192041064242
```

This indicates that bundle of conv1 -> conv5 is assigned to Jetson Nano, with a fixed GPU frequency of 691 MHz. The bundle of fc1 -> fc3 is assigned to wiscad cloud device. The overall latency is 0.067 seconds, with 0.016 s from computation and 0.05 from communication. Total energy consumed on the Jetson Nano is 91.6 mJ, with 89.5 mJ coming from computation on Nano and 2.0 mJ coming from communication.


## Special Considerations
At the time of publication, the Jetson Nano supported Jetpack 4.6. The NVIDIA-built releases of PyTorch are limited to Python 3.6 and PyTorch 1.11. This informs some of the decisions on supported libraries and methods, in particular compatible versions of Pandas used on the Jetson Nano during profiling. It may limit the ability to use newer networks. For instance, to utilize the ViT network, we backported the code from a newer version of torchvision and made minor adjustments where it used newer features not supported by PyTorch 1.11. This is shown in [torchvision_mod](torchvision_mod).

## Acknowledgement
This material is based upon work supported by the National Science Foundation under Grant No. 2006394.

## References


1. Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. 2012. ImageNet Clas-
sification with Deep Convolutional Neural Networks. In Advances in Neural
Information Processing Systems, F. Pereira, C. J. C. Burges, L. Bottou, and K. Q.
Weinberger (Eds.), Vol. 25.
2. Karen Simonyan and Andrew Zisserman. 2014. Very Deep Convolutional Net-
works for Large-Scale Image Recognition. https://arxiv.org/abs/1409.1556
3. Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xi-
aohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg
Heigold, Sylvain Gelly, et al . 2020. An image is worth 16x16 words: Transformers
for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020)
4. Amir Erfan Eshratifar, Mohammad Saeed Abrishami, and Massoud Pedram. 2021.
JointDNN: An Efficient Training and Inference Engine for Intelligent Mobile
Cloud Computing Services. IEEE Transactions on Mobile Computing 20, 2 (2021),
565–576.