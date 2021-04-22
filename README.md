## RSG: A Simple but Effective Module for Learning Imbalanced Datasets (CVPR 2021)

A Pytorch implementation of CVPR 2021 paper "RSG: A Simple but Effective Module for Learning Imbalanced Datasets". RSG (Rare-class Sample Generator) is a flexible module that can generate rare-class samples during training and can be combined with any backbone network. RSG is only used in the training phase, so it will not bring any additional burdens to the backbone network in the testing phase.


How to use RSG in your own networks
-----------------
1. Initialize RSG module:

   ```
   from RSG import *

   # n_center: The number of centers, e.g., 15.
   # feature_maps_shape: The shape of input feature maps (channel, width, height), e.g., [32, 16, 16].
   # num_classes: The number of classes, e.g., 10.
   # contrastive_module_dim: The dimention of the contrastive module, e.g., 256.
   # head_class_lists: The index of head classes, e.g., [0, 1, 2].
   # transfer_strength: Transfer strength, e.g., 1.0.
   # epoch_thresh: The epoch index when rare-class samples are generated: e.g., 159.

   self.RSG = RSG(n_center = 15, feature_maps_shape = [32, 16, 16], num_classes=10, contrastive_module_dim = 256, head_class_lists = [0, 1, 2], transfer_strength = 1.0, epoch_thresh = 159)

   ```

2. Use RSG in the forward pass during training:

   ```
   out = self.layer2(out)

   # feature_maps: The input feature maps.
   # head_class_lists: The index of head classes.
   # target: The label of samples.
   # epoch: The current index of epoch.

   if phase_train == True:
     out, cesc_total, loss_mv_total, combine_target = self.RSG.forward(feature_maps = out, head_class_lists = [0, 1, 2], target = target, epoch = epoch)
    
   out = self.layer3(out) 
   ```

The two loss terms, namely ''cesc_total'' and ''loss_mv_total'', will be returned and combined with cross-entropy loss for backpropagation. More examples and details can be found in the models in the directory ''Imbalanced_Classification/models''.

How to train the model
-----------------
Some examples:

Go into the "Imbalanced_Classification" directory.

1. To reimplement the result of ResNet-32 on long-tailed CIFAR-10 ($\rho$ = 100) with RSG and LDAM-DRW:

   ```
   Export CUDA_VISIBLE_DEVICES=0,1
   python cifar_train.py --imb_type exp --imb_factor 0.01 --loss_type LDAM --train_rule DRW
   ```

2. To reimplement the result of ResNet-32 on step CIFAR-10 ($\rho$ = 50) with RSG and Focal loss:

   ```
   Export CUDA_VISIBLE_DEVICES=0,1
   python cifar_train.py --imb_type step --imb_factor 0.02 --loss_type Focal --train_rule None
   ```

3. To run experiments on iNaturalist 2018, Places-LT, or ImageNet-LT:

   Firstly, please prepare datasets and their corresponding list files. For the convenience, we provide the list files in the Google Drive and Baidu Disk. 

   <table><tbody>
   <!-- START TABLE -->
   <!-- TABLE HEADER -->
   <th valign="bottom">Google Drive</th>
   <th valign="bottom">Baidu Disk</th>
   <!-- TABLE BODY -->
   <tr>
   <td align="center"><a href="https://drive.google.com/file/d/1EjcTqoJMbj6EfvY-yt1eaeMdHzSYBCy-/view?usp=sharing">download</a></td>
   <td align="center"><a href="https://pan.baidu.com/s/1gwP6qT9r_N7834IBaf4WTA">download</a>  (code: w63j)  </td>
   </tr>
   </tbody></table>

   To train the model:

   ```
   python inaturalist_train.py
   ```
   or
   
   ```
   python places_train.py
   ```
   or
   
   ```
   python imagenet_lt_train.py
   ```

   As for Places-LT or ImageNet-LT, the model is trained on the training set, and the best model on the validation set will be saved for testing.
   The "places_test.py" and 'imagenet_lt_test.py' are used for testing.

Citation
-----------------

```
@inproceedings{Jianfeng2021RSG,
  title = {RSG: A Simple but Effective Module for Learning Imbalanced Datasets},
  author = {Jianfeng Wang and Thomas Lukasiewicz and Xiaolin Hu and Jianfei Cai and Zhenghua Xu},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```
