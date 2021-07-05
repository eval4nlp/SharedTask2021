# SharedTask2021

This repository contains the data, baselines and evaluation scripts for the Eval4NLP Shared Task on
Explainable Quality Estimation.

- **Shared task official website**: https://eval4nlp.github.io/sharedtask.html
- **Repository**: https://github.com/eval4nlp/SharedTask2021
- **Submission website**: https://competitions.codalab.org/competitions/33038

## Datasets

### Training and development data

The directory `data` contains training and development data for Romanian-English (Ro-En) and Estonian-English (Et-En)
language pairs.

The directories for each data partition and language pair contain the following files:

- `<partition>.src`: source sentences
- `<partition>.mt`: MT outputs
- `<partition>.pe`: post-editing of the MT outputs
- `<partition>.da`: sentence-level quality scores
- `<partition>.hter`: HTER score
- `<partition>.tgt-tags`: word-level labels indicating whether each token in the MT output is an error (1)
  or is correct (0)
- `<partition>.src-tags`: word-level labels indicating whether each token in the source corresponds to an error (1)
  or to a correct token (0) in the MT output

All the data is tokenized. Word-level labels were derived by comparing the MT outputs with their post-edited versions
based on the alignments provided by the TER tool.

This data is an adjusted version of the [MLQE-PE dataset](https://github.com/sheffieldnlp/mlqe-pe) that was used
at the [WMT2020 Shared Task on Quality Estimation](http://www.statmt.org/wmt20/quality-estimation-task.html).
The differences with the QE Shared Task are as follows:
- For simplicity, gaps are ignored. Thus, the number of word-level labels corresponds to the number of tokens.
- OK and BAD labels are replaced by 0 and 1, respectively.

We provide a gold standard of 20 annotated sentence pairs for DE-ZH and RU-DE. This may be used for the participants to gain an intuition of the task for those languages: [DE-ZH](https://drive.google.com/file/d/10Pe5mURpCnU4f22DNWErr4jj-5UmvaFx/view?usp=sharing), [RU-DE](https://drive.google.com/file/d/1IAeT-LOUDWlN3vTQaTgnxQdXmt7_fAR6/view?usp=sharing). <!-- The data is not tokenized in this case, unlike the data used eventually in the shared task (further,  the different colors codings are not relevant). -->


### Test data

As test data, we are collecting sentence-level quality scores and word-level error annotations for Et-En and Ro-En,
as well as two zero-shot language pairs: German-Chinese (DE-ZH) and Russian-German (RU-DE). Human annotators are 
asked to indicate translation errors as an explanation for the overall sentence score, as well as the corresponding words
in the source sentence. The guidelines for this annotation effort are available in the `annotation-guidelines` directory.


## Baselines

We provide the following baselines:

* [TransQuest-LIME](baselines/transquest-lime.ipynb): This method uses the pre-trained sentence-level
quality estimation models available from the [TransQuest toolkit](https://github.com/TharinduDR/TransQuest)
to rate the translations, and uses the [LIME explainer](https://github.com/marcotcr/lime) to explain the ratings.
* [XMover-SHAP](baselines/xmover-shap/xmover-shap.ipynb): This method uses [XMover](https://github.com/AIPHES/ACL20-Reference-Free-MT-Evaluation)
to rate translations and uses the [SHAP explainer](https://github.com/slundberg/shap) to explain the ratings.


## Evaluation

The following metrics will be used to assess performance:
- AUC score
- AP (Average Precision)
- Recall at top-K

To run evaluation on a toy example:
```
cd scripts
python evaluate.py --gold_explanations_fname example/target-gold.example.roen 
                   --model_explanations_fname example/target.example.roen
                   --gold_sentence_scores_fname example/sentence-gold.example.roen
                   --model_sentence_scores_fname example/sentence.example.roen
```

## Contact information
- Website: https://eval4nlp.github.io/
- Email: eval4nlp@gmail.com
