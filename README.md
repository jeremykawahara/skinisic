# skinisic
`skinisic` provides a trained convolutional neural network to detect the presence of four dermoscopic criteria (pigment network, negative network, milia-like cysts, streaks) within a dermoscopy image.

Our entry had the highest AUROC score on the 
<a href="https://challenge.kitware.com/#challenge/583f126bcad3a51cc66c8d9a">ISBI ISIC 2017 Challenge</a> Part 2 - Lesion Dermoscopic Feature Extraction, and ranked <a href="https://challenge.kitware.com/#phase/584b0afacad3a51cc66c8e2e">first place</a> on this task.

The CNN model provided achieves a higher Jaccard Index than the CNN entry used in the challenge. We make the case that the Jaccard Index provides a more clinically meaningful measure of performance than the AUROC score. More details can be found in,

> J. Kawahara and G. Hamarneh, “Fully convolutional neural networks to detect clinical dermoscopic features,” IEEE J. Biomed. Heal. Informatics, vol. 23, no. 2, pp. 578–585, 2019. https://doi.org/10.1109/JBHI.2018.2831680 [<a href="https://arxiv.org/pdf/1703.04559.pdf">PDF</a>]

# Installation
`skinisic` is a Python module that relies on <a href="https://keras.io/">Keras</a>.

You can see the dependencies and versions tested on <a href="https://github.com/jeremykawahara/skinisic/blob/master/version_check.ipynb">here</a>.

To use `skinisic`,
  1. Download the trained model.</li>
  2. Clone this repository to your local machine.</li>
  `git clone https://github.com/jeremykawahara/skinisic.git`
  3. Run the minimal example.</li>

