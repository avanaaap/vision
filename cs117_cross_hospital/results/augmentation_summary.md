| Augmentation | AdaBN | n | CheXpert test macro | NIH test macro | NIH cardiomegaly | NIH pleural_effusion | NIH pneumothorax | Drop |
|---|---|---|---|---|---|---|---|---|
| baseline | OFF | 3 | 0.894 ± 0.003 | 0.753 ± 0.013 | 0.757 ± 0.016 | 0.757 ± 0.011 | 0.746 ± 0.021 | 0.141 ± 0.015 |
| baseline | ON | 3 | 0.894 ± 0.003 | 0.750 ± 0.018 | 0.743 ± 0.028 | 0.750 ± 0.012 | 0.758 ± 0.019 | 0.144 ± 0.020 |

Mean ± sample std over seeds. CheXpert test is always scored with source BN statistics; Drop = CheXpert internal test macro-AUROC − NIH test macro-AUROC.
