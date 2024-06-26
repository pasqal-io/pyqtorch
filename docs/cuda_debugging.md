To debug your quantum programs on `CUDA` devices, `pyqtorch` offers a `DEBUG` mode, which can be activated via
setting the `PYQ_LOG_LEVEL` environment variable.

```bash
export PYQ_LOG_LEVEL=DEBUG
```

Before running your script, make sure to install the following packages:

```bash
pip install nvidia-pyindex
pip install nvidia-dlprof[pytorch]
```
For more information, check [the dlprof docs](https://docs.nvidia.com/deeplearning/frameworks/dlprof-user-guide/index.html).
