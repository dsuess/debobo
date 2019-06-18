D.E. Bobo - Detection Evaluation for Bounding Boxes
===================================================

<div style="text-align:center">
  <img src="https://raw.githubusercontent.com/dseuss/debobo/2c5e651ff89d1c189a2a33ef4857061bf9eb7e6a/assets/the_dj.jpg" width="360">
</div>

DEBobo is a library providing an easy-to-use evaluation code for object detection models. 
It's main motivation was to replace the part of [pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools) responsible for evaluation as it doesn't work well with custom datasets.
Additionally, I found the workflow of pycocotools doesn't work well with high-level training libraries such as [ignite](https://github.com/pytorch/ignite). 


## Installation

Most end users should be able to get away with

```
pip install debobo
```

For development, install the library with symlinks and with the additional test requirements using

```
pip install -e .[test]
```

To run the tests, first download the test-data using `./fetch_testdata.sh`. 
The test is run via 

```
pytest tests/
```

It compares the results obtained from debobo to the result obtained using pycocotools.


## Usage

We provide ready-to-use metrics for ignite in `debobo.adapters.ignite`. 
Feel free to request other adapters as an issue.
Also, check the [tests](tests/test_detection.py) for how to use the low-level routines.


## Thanks

Thanks to [@martiningram](https://github.com/martiningram/) for the header image.
