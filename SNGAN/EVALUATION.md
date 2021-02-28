# Environment
```shell
python==3.6
pytorch==1.4.0 
tensorflow==1.15
imageio
numpy
```

Install command:
```python
conda install pytorch==1.4.0 cudatoolkit=10.0 -c pytorch # cuda101
pip install tensorflow-gpu==1.15 # tensorflow-gpu for py36
pip install imageio tqdm scikit-learn
```
# Command
```python
python evaluation.py --load-path checkpoint.pth(change this)
```

In `evaluation.py` the model will be pruned using `torch.nn.utils.prune.custom_from_mask` function with mask from trained model in `checkpoint.pth`. 