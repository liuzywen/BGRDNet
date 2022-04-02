# BGRDNet: RGB-D Salient Object Detection with a Bidirectional Gated Recurrent Decoding Network
The paper has been published in Multimedia Tools and Applications.[paper]()


##Abstract
Traditional U-Net framework generates multi-level features by the successive convolution and pooling operations, and then decodes the saliency cue by progressive upsampling and skip connection. The multi-level features are generated from the same input source, but quite different with each other. In this paper, we explore the complementarity among multi-level features, and decode them by Bi-GRU. Since multi-level features are different in the size, we first propose scale adjustment module to organize multi-level features into sequential data with the same channel and resolution. The core unit SAGRU of Bi-GRU is then devised based on self-attention, which can effectively fuse the history and current input. Based on the designed SAGRU, we further present the bidirectional decoding fusion module,
which decoding the multi-level features in both down-top and top-down manners. The proposed bidirectional gated recurrent decoding network is applied in the RGB-D salient object detection, which leverages the depth map as a complementary information. Concretely, we put forward depth guided residual module to enhance the color feature. Experimental results demonstrate our method outperforms the state-of-the-art methods in the six popular benchmarks. Ablation studies also verify each module plays an important role.

## Pretraing 

链接：https://pan.baidu.com/s/1bxO0ygO3VXUjPXntleVNtg 
提取码：b9yd 



## Training Set
2185
https://drive.google.com/file/d/1fcJj4aYdJ6N-TvvxSZ_sBo-xhtd_w-eJ/view?usp=sharing


2985
https://drive.google.com/file/d/1mYjaT_FTlY4atd-c0WdQ-0beZIpf8fgh/view?usp=sharing

##  Result Saliency Maps
链接：
提取码： 



### Citation

If you find the information useful, please consider citing:

```
@article{liu2022bgrdnet,
  title={BGRDNet: RGB-D salient object detection with a bidirectional gated recurrent decoding network},
  author={Liu, Zhengyi and Wang, Yuan and Zhang, Zhili and Tan, Yacheng},
  journal={Multimedia Tools and Applications},
  pages={1--21},
  year={2022},
  publisher={Springer}
}
```
