# SharedTask2021

This repository contains the data, baselines and evaluation scripts for the Eval4NLP Shared Task on
Explainable Quality Estimation.

- **Shared task official website**: https://eval4nlp.github.io/sharedtask.html
- **Repository**: https://github.com/eval4nlp/SharedTask2021
- **Submission website**: https://competitions.codalab.org/competitions/33038

## Datasets

### Training and development data (for sentence-level scores)

The directory `data` contains training and development data for Romanian-English (Ro-En) and Estonian-English (Et-En)
language pairs.

Especially, the data contains _sentence-level_ scores and _word-level scores_. 

   - The sentence-level scores can be used to train a supervised model indicating the similarity  between source and target sentences. If participants wish, they can also use unsupervised approaches like XMoverScore, i.e., they can ignore the sentence-level training scores. 

   - The data also contains _word-level labels_. Word-level labels were derived by comparing the MT outputs with their post-edited versions
based on the alignments provided by the TER tool. 

   - The word-level labels can be used for **exploratory purposes**, to estimate how well a model performs for word-level explainability (in the absence of test data, which will be provided only in a later stage of the shared task). If participants don't train their systems on word-level labels, they will be routed to the **constrained track**.

   - Alternatively, participants may train a system on the existing word-level annotations (**unconstrained track**). In this case, participants should be aware that the human annotations may be similar to the word-level annotations provided below in the test data, but there will not be a full correspondence (similar to a domain shift).

Participants submitting in the constrained tracks will be evaluated separately from participants submitting in the unconstrained track.

<!--
**Note:** 

> The sentence-level QE systems can be trained using sentence-level quality scores. Word-level labels derived from post-editing can be used for development purposes. However, we **discourage** participants from using the word-level data for training, as the goal of the shared task is to explore word-level quality estimation in an unsupervised setting, i.e. as a rationale extraction task.

The directories for each data partition and language pair contain the following files:
-->

**Data format**

- `<partition>.src`: source sentences
- `<partition>.mt`: MT outputs
- `<partition>.pe`: post-editing of the MT outputs
- `<partition>.da`: sentence-level quality scores
- `<partition>.hter`: HTER score
- `<partition>.tgt-tags`: word-level labels indicating whether each token in the MT output is an error (1)
  or is correct (0)
- `<partition>.src-tags`: word-level labels indicating whether each token in the source corresponds to an error (1)
  or to a correct token (0) in the MT output

All the data is tokenized. 

This data is an adjusted version of the [MLQE-PE dataset](https://github.com/sheffieldnlp/mlqe-pe) that was used
at the [WMT2020 Shared Task on Quality Estimation](http://www.statmt.org/wmt20/quality-estimation-task.html).
The differences with the QE Shared Task are as follows:
- For simplicity, gaps are ignored. Thus, the number of word-level labels corresponds to the number of tokens.
- OK and BAD labels are replaced by 0 and 1, respectively.

**Additionally**:

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
* [XMover-SHAP](baselines/xmover-shap/xmover-shap-el-en.ipynb): This method uses [XMover](https://github.com/AIPHES/ACL20-Reference-Free-MT-Evaluation)
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


## Citation
```
@inproceedings{fomicheva-etal-2021-eval4nlp,
    title = "The {E}val4{NLP} Shared Task on Explainable Quality Estimation: Overview and Results",
    author = "Fomicheva, Marina  and
      Lertvittayakumjorn, Piyawat  and
      Zhao, Wei  and
      Eger, Steffen  and
      Gao, Yang",
    booktitle = "Proceedings of the 2nd Workshop on Evaluation and Comparison of NLP Systems",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.eval4nlp-1.17",
    pages = "165--178",
}
```

## Contact information
- Website: https://eval4nlp.github.io/
- Email: eval4nlp@gmail.com
