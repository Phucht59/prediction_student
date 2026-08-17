# Student Academic Risk System

Current module: Prediction

Final model: Hybrid CNN + BiLSTM (`model_id=hybrid`, `display_name=Hybrid`)

Datasets: UCI and OULAD

- OULAD FINAL-100 is the principal prediction result.
- OULAD 20/35/50/75 are supporting early-warning evaluations.
- UCI full-information is the principal result; S0/S1/S2 are supporting early-prediction evaluations.

Future module: Recommendation

This project is a byte-preserving extraction of the final prediction code and evidence. No training or outer evaluation is performed here.

Checkpoint note: the Phase8 outer run produced saved predictions/results but no Phase8 final model checkpoint. The copied runtime therefore requires an authorized Phase8 checkpoint before standalone final inference can be enabled; no older checkpoint is substituted.
