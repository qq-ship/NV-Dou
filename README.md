NV-DOU is a kind of doudizhu algorithm based on reinforcement learning, and its playing environment is based on RLCard.
* RLCard site: [http://rlcard.org](http://rlcard.org)
* The original code of RLCard code URL:[https://github.com/datamllab/rlcard](https://github.com/datamllab/rlcard)  
We have modified RLcard, so you need to use the RLcard code we provide to run NV-Dou. RLCard can be installed in the same way as before
## Installation and Running
Make sure that you have **Python 3.68+** and **pip** installed. We recommend installing `rlcard` and run 'NV-Dou' with `pip` as follow:
RLCard:
```
download the rlcard that we provide
cd rlcard
pip install -e .
```
NV-Dou:
```
download the source_code that we provide
cd source_code
python main_a3c.py(This code must be the GPU version of PyTorch, and you'll need to prepare the PyTorch's GPU environment)
```


