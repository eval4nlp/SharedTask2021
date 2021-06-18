# Baselines

We provide the following baselines for the shared task:

* [TransQuest-LIME](./transquest-lime.ipynb): This method uses the pre-trained sentence-level
quality estimation models available from the [TransQuest toolkit](https://github.com/TharinduDR/TransQuest)
to rate the translations, and uses the [LIME explainer](https://github.com/marcotcr/lime) to explain the ratings.
* [XMover-SHAP](./xmover-shap/xmover-shap-et-en.ipynb): This method uses [XMover](https://github.com/AIPHES/ACL20-Reference-Free-MT-Evaluation)
to rate translations and uses the [SHAP explainer](https://github.com/slundberg/shap) to explain the ratings.

**Pytorch** is required by both baseline methods. The recommended version of PyTorch is 1.8. Please refer to [PyTorch installation page](https://pytorch.org) for the specific installation command for your platform.

## Performance

* **RO-EN**(Romanian to English)

| Methods | AUC | AP | Recall at Top-K |
|---------|-----|----|-----------------|
| TransQuest-LIME | .353 | .300 | .310 |
| XMover-SHAP | .638 | .464 | .339 |


* **ET-EN**(Estonian to English)

| Methods | AUC | AP | Recall at Top-K |
|---------|-----|----|-----------------|
| XMover-SHAP | .583 | .456 | .352 |

