# mare
Repository for the paper submission


# Environment

Tested with Ubuntu 18.04, Anaconda 2020.11 and NVIDIA driver version 450.102.04

If you have no coda-compatible GPU, delete the cudatoolkit dependency 
from the `environment.yml` file.

To install the conda environment execute
```shell
conda env create -f environment.yml
```

This may take several minutes.

# Reproduction of results

To reproduce the values from the Paper, download the archive

The following instructions can be used to reproduce the results in the paper.

## Sequence Tagging

The values for AR, Cl, MRE, CRE und BRE correspond to the values of 
**MARE Seq. Tag.** in **Table 2**.

The F1 scores for AR_no_trigger und MRE_no_trigger correspond to the values
of **Seq. Tag. with Trigger** un **Table 3**

```shell
sh scripts/evaluate_model.sh models/sequence.tar.gz evaluations/seq_tag seq_lab_elmo_pred
```

The result shoud be

```
EVALUATION RESULTS FOR MRE

precision_micro: 0.420689655172413
recall_micro: 0.4765625
f1_micro: 0.446886446886446

EVALUATION RESULTS FOR Cl

precision_micro: 0.7103448275862061
recall_micro: 0.8046875
f1_micro: 0.754578754578754

EVALUATION RESULTS FOR CRE

precision_micro: 0.268965517241379
recall_micro: 0.3046875
f1_micro: 0.28571428571428503

EVALUATION RESULTS FOR AR

precision_micro: 0.6567164179104471
recall_micro: 0.6591760299625461
f1_micro: 0.6579439252336441

EVALUATION RESULTS FOR BRE

precision_micro: 0.435185185185185
recall_micro: 0.49473684210526303
f1_micro: 0.46305418719211805

EVALUATION RESULTS FOR MRE_no_trigger

precision_micro: 0.462068965517241
recall_micro: 0.5234375
f1_micro: 0.49084249084249004

EVALUATION RESULTS FOR AR_no_trigger

precision_micro: 0.633251833740831
recall_micro: 0.6301703163017031
f1_micro: 0.6317073170731701
```

## Span Labeling

The values for AR, Cl, MRE, CRE und BRE correspond to the values of 
**MARE Span Lab.** in **Table 2**.

The F1 scores for AR_no_trigger und MRE_no_trigger correspond to the values
of **Span Lab. with Trigger** un **Table 3**

```shell
sh scripts/evaluate_model.sh models/span_based.tar.gz evaluations/span_lab mare.span_based_precidtor.SpanBasedPredictor
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
sh scripts/evaluate_model.sh models/dygiepp.tar.gz evaluations/dygiepp mare.evaluation.mock_model.DygieppMockModel
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

## SpERT

The value for BRE corresponds to the values of 
**SpERT** in **Table 2**.

```shell
sh scripts/evaluate_model.sh models/span_based.tar.gz evaluations/test mare.span_based_precidtor.SpanBasedPredictor
```

The result shoud be

```

```

## Sequence Tagging Baseline
The values for AR, Cl, MRE, CRE und BRE correspond to the values of 
**MARE Baseline** in **Table 2**.


```shell
sh scripts/evaluate_model.sh models/sequence_tagging_baseline.tar.gz evaluations/seq_tag_baseline seq_lab_elmo_pred
```

The result shoud be

```
EVALUATION RESULTS FOR MRE

precision_micro: 0.389312977099236
recall_micro: 0.3984375
f1_micro: 0.39382239382239304

EVALUATION RESULTS FOR Cl

precision_micro: 0.65648854961832
recall_micro: 0.671875
f1_micro: 0.6640926640926641

EVALUATION RESULTS FOR CRE

precision_micro: 0.24427480916030503
recall_micro: 0.25
f1_micro: 0.24710424710424703

EVALUATION RESULTS FOR AR

precision_micro: 0.6561797752808981
recall_micro: 0.5468164794007491
f1_micro: 0.5965270684371801

EVALUATION RESULTS FOR BRE

precision_micro: 0.408163265306122
recall_micro: 0.421052631578947
f1_micro: 0.41450777202072503

EVALUATION RESULTS FOR MRE_no_trigger

precision_micro: 0.427480916030534
recall_micro: 0.4375
f1_micro: 0.432432432432432

EVALUATION RESULTS FOR AR_no_trigger

precision_micro: 0.621951219512195
recall_micro: 0.49635036496350304
f1_micro: 0.552097428958051

```

## Sequence Tagging No Trigger
The F1 scores for AR_no_trigger und MRE_no_trigger correspond to the values
of **Seq. Tag. without Trigger** in **Table 3**


```shell
sh scripts/evaluate_model.sh models/sequence_no_trigger.tar.gz evaluations/seq_tag_no_trig seq_lab_elmo_pred
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
sh scripts/evaluate_model.sh models/span_based_no_trigger_local.tar.gz evaluations/span_lab_no_trig mare.span_based_precidtor.SpanBasedPredictor
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

