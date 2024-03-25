# Install
## Requirements

* CUDA>=11.8
  
* Python>=3.9
  
  Create Python environments.
  ```bash
  conda create -n omdet python=3.9
  ```
  Activate the environment:
  ```bash
  conda activate omdet
  ```

* Pytorch>=2.1.0, Torchvision>=0.17.1
  
  If your CUDA version is 11.8, you can install Pytorch as following:
  ```bash
  conda install pytorch==2.1.0 torchvision==0.17.1 pytorch-cuda=11.8 -c pytorch -c nvidia
  ```

* detectron2>=0.6.0:

  Install detectron2:
  ```bash
  python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
  ```

* Other requirements
    ```bash
    pip install -r requirements.txt
    ```