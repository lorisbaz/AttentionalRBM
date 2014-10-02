AttentionalRBM
==============
This code implements a tracker with binary RBM observation model combined with an region selection model based on reinforcement learning algorithms. Here, we provide the code to do the experiments for the single target synthetic experiment reported in our paper. 

**Learning attentional policies for object tracking and recognition in video with deep networks**,
L. Bazzani, N. de Freitas, H. Larochelle, V. Murino, and J-A Ting,
*In International Conference on Machine Learning (ICML), 2011.*

##INSTRUCTIONS
In order to run the experiments using this package, you should follow these steps:

1. Train the binary RBM on soft data with `binRBM_GAZE_train_allMNIST.m`
2. Train the multiclass logistic classifier `logreg_GAZE_binRBM_MNIST_all.m`
3. Build your own video using `build_synth_dataset.m`
  or you can use the provided videos at http://www.lorisbazzani.info/datasets/ICML11_synth_dataset.zip
4. Run the tracking script `main_synth_GAZE_RBMtracking.m`
5. Watch the final results `display_final_res.m` at the end of the tracking script
6. Compare the accuracy of tracking between the three policies in several experiments `error_estimation.m`

##3RD-PART LIBRARIES
We provide the binary Restricted Bolzmann Machine library written by Kevin Swersky.

##REFERENCE AND ACKNOWLEDGEMENTS
  ```
  @inproceedings{Bazzani:2011ICML,
  author = {Bazzani, Loris and de Freitas, Nando and Larochelle, Hugo, and Murino, Vittorio and Ting, Jo-Anne},
  title = {Learning attentional policies for tracking and recognition in video with deep networks},
  booktitle = {International Conference on Machine Learning},
  year = {2011},
  address = {Seattle, WA, United States},
  }
  ```

The code has been written during the period I spent at UBC, British Columbia, Vancouver, under the supervision of Prof. Nando de Freitas.
We thank Kevin Swersky for providing the code for binary RBMs' training and testing.
