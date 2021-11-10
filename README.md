# TinyNeRF PyTorch C++ Renderer

The [NeRF code](https://github.com/bmild/nerf) release has an accompanying [Colab notebook](https://colab.research.google.com/github/bmild/nerf/blob/master/tiny_nerf.ipynb), that showcases training a feature-limited version of NeRF on a "tiny" scene.

This PyTorch C++ port is based on [krrish94](https://github.com/krrish94)´s PyTorch [reimplementation](https://colab.research.google.com/drive/1rO8xo0TemN67d4mTpakrKrLp03b9bgCX) of the notebook.

Components not included in the notebook

- 5D input including view directions
- Hierarchical Sampling

Known issues:
- Out of memory when trying to render in a loop
 
