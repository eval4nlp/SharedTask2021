# Baselines

We provide the following baselines for the shared task:

* [TransQuest-LIME](./transquest-lime.ipynb): This method uses the pre-trained sentence-level
quality estimation models available from the [TransQuest toolkit](https://github.com/TharinduDR/TransQuest)
to rate the translations, and uses the [LIME explainer](https://github.com/marcotcr/lime) to explain the ratings.
* [XMover-SHAP](./xmover-shap/xmover-shap.ipynb): This method uses [XMover](https://github.com/AIPHES/ACL20-Reference-Free-MT-Evaluation)
to rate translations and uses the [SHAP explainer](https://github.com/slundberg/shap) to explain the ratings.

**Pytorch** is required by both baseline methods. The recommended version of PyTorch is 1.8. Please refer to [PyTorch installation page](https://pytorch.org) for the specific installation command for your platform.
