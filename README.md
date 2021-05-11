# MARE - Multi-Attribute Relation Extraction
Repository for the paper submission: #TODO: insert link, when available


# Environment

Tested with Ubuntu 18.04, Anaconda 2020.11 and NVIDIA driver version 450.102.04
If you have a lower driver version and you don't need to train own models,
we recommend to install the environment without the cuda requirement.

If you have no coda-compatible GPU, delete the cudatoolkit dependency 
from the `environment.yml` file. If you do not have a cuda GPU, remove the line
```
- cudatoolkit=10.2.89
```
from `environment.yml`.

To install the conda environment execute
```shell
conda env create -f environment.yml
```

Install mare (the local directory) via pip

```shell
pip install -e .
```

This may take several minutes.

# Reproduction of results

To reproduce the values from the Paper, download the corresponding models
from https://fh-aachen.sciebo.de/s/D5FLVN7qk2UTCmX and put the `.tar.gz`
files in the models folder or execute the following shell commands.

```shell
wget -c https://fh-aachen.sciebo.de/s/D5FLVN7qk2UTCmX/download -O models.zip

unzip models.zip

rm models.zip
```

The following instructions can be used to reproduce the results in the paper.
All evaluations create a subfolder in evaluations.

## Sequence Tagging

The values for AR, Cl, MRE, CRE und BRE correspond to the values of 
**MARE Seq. Tag.** in **Table 2**.

The F1 scores for AR_no_trigger und MRE_no_trigger correspond to the values
of **Seq. Tag. with Trigger** un **Table 3**

```shell
sh scripts/evaluate_model.sh models/sequence.tar.gz evaluations/seq_tag seq_lab_elmo_pred mare
```

The result shoud be

```
EVALUATION RESULTS FOR MRE

precision_micro: 0.42957746478873204
recall_micro: 0.4765625
f1_micro: 0.45185185185185106

EVALUATION RESULTS FOR Cl

precision_micro: 0.725352112676056
recall_micro: 0.8046875
f1_micro: 0.7629629629629631

EVALUATION RESULTS FOR CRE

precision_micro: 0.28169014084507005
recall_micro: 0.3125
f1_micro: 0.296296296296296

EVALUATION RESULTS FOR AR

precision_micro: 0.660412757973733
recall_micro: 0.6591760299625461
f1_micro: 0.659793814432989

EVALUATION RESULTS FOR BRE

precision_micro: 0.439252336448598
recall_micro: 0.49473684210526303
f1_micro: 0.46534653465346504

EVALUATION RESULTS FOR MRE_no_trigger

precision_micro: 0.464788732394366
recall_micro: 0.515625
f1_micro: 0.48888888888888804

EVALUATION RESULTS FOR AR_no_trigger

precision_micro: 0.6410891089108911
recall_micro: 0.6301703163017031
f1_micro: 0.635582822085889
```

## Span Labeling

The values for AR, Cl, MRE, CRE und BRE correspond to the values of 
**MARE Span Lab.** in **Table 2**.

The F1 scores for AR_no_trigger und MRE_no_trigger correspond to the values
of **Span Lab. with Trigger** un **Table 3**

```shell
sh scripts/evaluate_model.sh models/span_based.tar.gz evaluations/span_lab mare.span_based_precidtor.SpanBasedPredictor mare
```

The result shoud be

```
EVALUATION RESULTS FOR MRE

precision_micro: 0.47244094488188904
recall_micro: 0.46875000000000006
f1_micro: 0.47058823529411703

EVALUATION RESULTS FOR Cl

precision_micro: 0.8031496062992121
recall_micro: 0.796875
f1_micro: 0.8

EVALUATION RESULTS FOR CRE

precision_micro: 0.291338582677165
recall_micro: 0.2890625
f1_micro: 0.290196078431372

EVALUATION RESULTS FOR AR

precision_micro: 0.751619870410367
recall_micro: 0.651685393258427
f1_micro: 0.698094282848545

EVALUATION RESULTS FOR BRE

precision_micro: 0.49473684210526303
recall_micro: 0.49473684210526303
f1_micro: 0.49473684210526303

EVALUATION RESULTS FOR MRE_no_trigger

precision_micro: 0.519685039370078
recall_micro: 0.515625
f1_micro: 0.517647058823529

EVALUATION RESULTS FOR AR_no_trigger

precision_micro: 0.7298850574712641
recall_micro: 0.618004866180048
f1_micro: 0.6693017127799731
```

## Dygie ++

The values for AR, Cl, MRE, CRE und BRE correspond to the values of 
**Dygie++** in **Table 2**.

The F1 scores for AR_no_trigger und MRE_no_trigger correspond to the values
of **Dygie++ with Trigger** in **Table 3**

```shell
sh scripts/evaluate_model.sh models/dygiepp.tar.gz evaluations/dygiepp mare.evaluation.mock_model.DygieppMockModel mare
```

The result shoud be

```
EVALUATION RESULTS FOR MRE

precision_micro: 0.47154471544715404
recall_micro: 0.453125
f1_micro: 0.46215139442231

EVALUATION RESULTS FOR Cl

precision_micro: 0.7723577235772351
recall_micro: 0.7421875
f1_micro: 0.7569721115537841

EVALUATION RESULTS FOR CRE

precision_micro: 0.260162601626016
recall_micro: 0.25
f1_micro: 0.254980079681274

EVALUATION RESULTS FOR AR

precision_micro: 0.630434782608695
recall_micro: 0.651685393258427
f1_micro: 0.6408839779005521

EVALUATION RESULTS FOR BRE

precision_micro: 0.550561797752809
recall_micro: 0.51578947368421
f1_micro: 0.5326086956521741

EVALUATION RESULTS FOR MRE_no_trigger

precision_micro: 0.536585365853658
recall_micro: 0.515625
f1_micro: 0.525896414342629

EVALUATION RESULTS FOR AR_no_trigger

precision_micro: 0.596810933940774
recall_micro: 0.6374695863746951
f1_micro: 0.616470588235294
```

## SpERT (SpART = SpERT with AllenNLP)

The value for BRE corresponds to the values of 
**SpERT** in **Table 2**.
```shell
sh scripts/evaluate_model.sh models/spart.tar.gz evaluations/spert spart spart
```

The result shoud be

```
EVALUATION RESULTS FOR MRE

precision_micro: 0.43269230769230704
recall_micro: 0.3515625
f1_micro: 0.387931034482758

EVALUATION RESULTS FOR Cl

precision_micro: 0.596153846153846
recall_micro: 0.484375
f1_micro: 0.5344827586206891

EVALUATION RESULTS FOR CRE

precision_micro: 0.08653846153846101
recall_micro: 0.0703125
f1_micro: 0.077586206896551

EVALUATION RESULTS FOR AR

precision_micro: 0.519230769230769
recall_micro: 0.202247191011235
f1_micro: 0.2911051212938

EVALUATION RESULTS FOR BRE

precision_micro: 0.573333333333333
recall_micro: 0.45263157894736805
f1_micro: 0.505882352941176

EVALUATION RESULTS FOR MRE_no_trigger

precision_micro: 0.48076923076923006
recall_micro: 0.390625
f1_micro: 0.43103448275862005

EVALUATION RESULTS FOR AR_no_trigger

precision_micro: 0.528
recall_micro: 0.16058394160583903
f1_micro: 0.246268656716417
```

## Sequence Tagging Baseline
The values for AR, Cl, MRE, CRE und BRE correspond to the values of 
**MARE Baseline** in **Table 2**.


```shell
sh scripts/evaluate_model.sh models/sequence_tagging_baseline.tar.gz evaluations/seq_tag_baseline seq_lab_elmo_pred mare
```

The result shoud be

```
EVALUATION RESULTS FOR MRE

precision_micro: 0.396825396825396
recall_micro: 0.390625
f1_micro: 0.39370078740157405

EVALUATION RESULTS FOR Cl

precision_micro: 0.682539682539682
recall_micro: 0.671875
f1_micro: 0.677165354330708

EVALUATION RESULTS FOR CRE

precision_micro: 0.26190476190476103
recall_micro: 0.2578125
f1_micro: 0.259842519685039

EVALUATION RESULTS FOR AR

precision_micro: 0.6591422121896161
recall_micro: 0.5468164794007491
f1_micro: 0.597748208802456

EVALUATION RESULTS FOR BRE

precision_micro: 0.40206185567010305
recall_micro: 0.410526315789473
f1_micro: 0.40625000000000006

EVALUATION RESULTS FOR MRE_no_trigger

precision_micro: 0.42857142857142805
recall_micro: 0.421875
f1_micro: 0.42519685039370003

EVALUATION RESULTS FOR AR_no_trigger

precision_micro: 0.6296296296296291
recall_micro: 0.49635036496350304
f1_micro: 0.5551020408163261

```

## Sequence Tagging No Trigger
The F1 scores for AR_no_trigger und MRE_no_trigger correspond to the values
of **Seq. Tag. without Trigger** in **Table 3**

Change the include_trigger Parameter in mare/seq_lab_elmo_pred.py to False.

```shell
sh scripts/evaluate_model.sh models/sequence_no_trigger.tar.gz evaluations/seq_tag_no_trig seq_lab_elmo_pred_no_trig mare
```

The result shoud be

```

EVALUATION RESULTS FOR MRE

precision_micro: 0.056
recall_micro: 0.0546875
f1_micro: 0.055335968379446

EVALUATION RESULTS FOR Cl

precision_micro: 0.728
recall_micro: 0.7109375
f1_micro: 0.7193675889328061

EVALUATION RESULTS FOR CRE

precision_micro: 0.048
recall_micro: 0.046875
f1_micro: 0.047430830039525

EVALUATION RESULTS FOR AR

precision_micro: 0.662337662337662
recall_micro: 0.47752808988764006
f1_micro: 0.554951033732317

EVALUATION RESULTS FOR BRE

precision_micro: 0.07865168539325801
recall_micro: 0.073684210526315
f1_micro: 0.07608695652173901

EVALUATION RESULTS FOR MRE_no_trigger

precision_micro: 0.512
recall_micro: 0.5
f1_micro: 0.50592885375494

EVALUATION RESULTS FOR AR_no_trigger

precision_micro: 0.662337662337662
recall_micro: 0.620437956204379
f1_micro: 0.6407035175879391
```

## Span Labeling No Trigger
The F1 scores for AR_no_trigger und MRE_no_trigger correspond to the values
of **Span Lab. without Trigger** in **Table 3**


```shell
sh scripts/evaluate_model.sh models/span_based_no_trigger_local.tar.gz evaluations/span_lab_no_trig mare.span_based_precidtor.SpanBasedPredictor mare
```

The result shoud be

```
EVALUATION RESULTS FOR MRE

precision_micro: 0.07563025210084001
recall_micro: 0.0703125
f1_micro: 0.072874493927125

EVALUATION RESULTS FOR Cl

precision_micro: 0.789915966386554
recall_micro: 0.734375
f1_micro: 0.761133603238866

EVALUATION RESULTS FOR CRE

precision_micro: 0.067226890756302
recall_micro: 0.0625
f1_micro: 0.064777327935222

EVALUATION RESULTS FOR AR

precision_micro: 0.72
recall_micro: 0.47191011235955005
f1_micro: 0.570135746606334

EVALUATION RESULTS FOR BRE

precision_micro: 0.103448275862068
recall_micro: 0.09473684210526301
f1_micro: 0.09890109890109801

EVALUATION RESULTS FOR MRE_no_trigger

precision_micro: 0.5630252100840331
recall_micro: 0.5234375
f1_micro: 0.542510121457489

EVALUATION RESULTS FOR AR_no_trigger

precision_micro: 0.72
recall_micro: 0.613138686131386
f1_micro: 0.6622864651773981

```
