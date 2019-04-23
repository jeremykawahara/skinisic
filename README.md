# skinisic
`skinisic` provides a trained convolutional neural network to detect the presence of four dermoscopic criteria (pigment network, negative network, milia-like cysts, streaks) within a dermoscopy image.

Our entry had the highest AUROC score on the 
<a href="https://challenge.kitware.com/#challenge/583f126bcad3a51cc66c8d9a">ISBI ISIC 2017 Challenge</a> Part 2 - Lesion Dermoscopic Feature Extraction, and ranked <a href="https://challenge.kitware.com/#phase/584b0afacad3a51cc66c8e2e">first place</a> on this task.

The CNN model provided achieves a higher Jaccard Index than the CNN entry used in the challenge. We make the case that the Jaccard Index provides a more clinically meaningful measure of performance than the AUROC score. More details can be found in,

> Jeremy Kawahara and Ghassan Hamarneh, “Fully convolutional neural networks to detect clinical dermoscopic features,” IEEE Journal of Biomedical and Health Informatics, vol. 23, no. 2, pp. 578–585, 2019. [<a href="https://doi.org/10.1109/JBHI.2018.2831680">DOI</a>] [<a href="https://arxiv.org/pdf/1703.04559.pdf">PDF</a>]

# Installation
`skinisic` is a Python module that relies on <a href="https://keras.io/">Keras</a>.

You can see the dependencies and versions tested on <a href="https://github.com/jeremykawahara/skinisic/blob/master/version_check.ipynb">here</a>.

To use `skinisic`,
  1. Navigate to your desired directory (e.g., `/projects` for this example) and open terminal.
  1. Clone this repository to your local machine:<br />
  `git clone https://github.com/jeremykawahara/skinisic.git`
  1. Download the trained model and save to disk (e.g,. `/projects/skinisic/notebooks/data`):<br />
  https://github.com/jeremykawahara/skinisic/releases/download/v0.0.1/vgg_f1-batch_aug.h5
  1. Navigate to the `skinisic` directory and run the minimal example (may take a seconds):<br />
  ```
  cd skinisic
  python minimal_example.py 'notebooks/data/isic2017-part2_vgg_f1-batch_aug.h5'
  ```
You should see the following output:
![Predicted Output](https://github.com/jeremykawahara/skinisic/blob/master/docs/figs/min_example_predicted.png)

You can find a more <a href="https://github.com/jeremykawahara/skinisic/blob/master/notebooks/isic2017_part2-detect-criteria_infer.ipynb">comprehensive example here</a>.

# Publications
If you find this code or model helpful, please consider citing our work:
```
@article{Kawahara2019,
author = {Kawahara, Jeremy and Hamarneh, Ghassan},
doi = {10.1109/JBHI.2018.2831680},
issn = {21682194},
journal = {IEEE Journal of Biomedical and Health Informatics},
number = {2},
pages = {578--585},
title = {Fully convolutional neural networks to detect clinical dermoscopic features},
volume = {23},
year = {2019}
}
```
