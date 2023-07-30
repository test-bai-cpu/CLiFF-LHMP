# CLiFF-LHMP

CLiFF-LHMP is a pattern-based human motion prediction approach. Provided with maps of dynamics, the method predicts human motion in a long term.

1. The following two datasets are supported:
- [ATC pedestrian tracking dataset](https://dil.atr.jp/ISL/crest2010_HRI/ATC_dataset/)
- [THÖR human motion trajectories dataset](http://thor.oru.se/): containing THÖR1 and THÖR3 with different environment setup.

Part of both datasets are put in `dataset` folder for demo. The dataset files provided are pre-processed. The downsample rate is 2.5 Hz. Maps are also included for visualizing the prediction results.


2. Two prediction approaches are supported:
- CLiFF-LHMP: use maps of dynamics for prediction
- CVM: use constant velocity model for prediction


3. To run the demo:
```
poetry run python main.py
```
