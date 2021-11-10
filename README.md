# TinyNeRF PyTorch C++ Renderer

The [NeRF code](https://github.com/bmild/nerf) release has an accompanying [Colab notebook](https://colab.research.google.com/github/bmild/nerf/blob/master/tiny_nerf.ipynb), that showcases training a feature-limited version of NeRF on a "tiny" scene.

This PyTorch C++ port is based on [krrish94](https://github.com/krrish94)´s PyTorch [reimplementation](https://colab.research.google.com/drive/1rO8xo0TemN67d4mTpakrKrLp03b9bgCX) of the original notebook.

Components not included in the notebook:
- 5D input including view directions
- Hierarchical Sampling

Dependencies:
- [PyTorch C++](https://pytorch.org/cppdocs/installing.html)
- [OpenCV](https://opencv.org/)
- [glm](https://github.com/g-truc/glm)

Known issues:
- Out of memory when trying to render in a loop
 

## Citation

```
@inproceedings{mildenhall2020nerf,
  title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
  author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
  year={2020},
  booktitle={ECCV},
}
```