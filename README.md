# CPPN Demo

[Compositional pattern-producing network](https://en.wikipedia.org/wiki/Compositional_pattern-producing_network) demo


## Installation

just install dependencies first
```
pip install -r requirements.txt
```

## Running
To create a set of abstract images in png format of size 128x128
```
python cppn.py --png
```

if you want any other size provide it as an argument, like
```
python cppn.py --png --size 640
```
and you will get an 640x640 image back


In case you want to play around with density and structure of a generated image then just provide *num_layers* and/or *num_neurons*, like
```
python cppn.py --png --num_neurons 20 --num_layers 20
```
or with a shortcuts:
```
python cppn.py --png -n 20 -l 20
```
