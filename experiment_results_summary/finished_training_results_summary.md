# Finished Training Results Summary

- Source: completed logs under `/home/yanghongsheng/rt-detr/train_log`.
- Inclusion rule: log contains `150 epochs completed` or `Optimizer stripped`. Runner, monitor and incomplete logs are excluded.
- Best metrics are selected by highest validation `mAP50-95` among parsed `all` rows.

## Grouped Summary

| Experiment/Config | Modules | Runs | Seeds | Best_mAP50-95_mean | Best_mAP50-95_std | Best_mAP50-95_max | Best_mAP50-95_min | Best_mAP50_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MSAInnov-P2Decoder | MSAInnov + P2 decoder branch | 1 | - | 0.505 | 0.000 | 0.505 | 0.505 | 0.831 |
| BCR beta=0.10 | BCRSODAttention(beta=0.10) | 3 | 0,1,2 | 0.500 | 0.001 | 0.501 | 0.499 | 0.834 |
| MSPConv | MSPConv | 1 | - | 0.500 | 0.000 | 0.500 | 0.500 | 0.834 |
| P2Detail | P2Detail module | 1 | - | 0.498 | 0.000 | 0.498 | 0.498 | 0.844 |
| BCR-DySample | BCRSODAttention + DySample | 3 | 0,1,2 | 0.497 | 0.005 | 0.502 | 0.490 | 0.829 |
| H2Net | H2Net modules | 3 | 0,-,- | 0.497 | 0.006 | 0.506 | 0.492 | 0.831 |
| MSA-SODAttention-Original | MSAOriginalSODAttention | 3 | 0,1,2 | 0.497 | 0.003 | 0.501 | 0.494 | 0.838 |
| HEMS-CSPMSEIE | CSPMSEIE / HEMS module | 1 | - | 0.497 | 0.000 | 0.497 | 0.497 | 0.822 |
| HFPG | HighFrequencyPriorGuidance | 1 | - | 0.496 | 0.000 | 0.496 | 0.496 | 0.822 |
| BCR beta=0.05 | BCRSODAttention(beta=0.05) | 3 | 0,1,2 | 0.495 | 0.006 | 0.502 | 0.488 | 0.827 |
| BCR-SPDDownsample | BCRSODAttention + SPDDownsample | 1 | 0 | 0.493 | 0.000 | 0.493 | 0.493 | 0.813 |
| BCR beta=0.20 | BCRSODAttention(beta=0.20) | 1 | 0 | 0.492 | 0.000 | 0.492 | 0.492 | 0.847 |
| SODGA-P2Detail | SODGuidedAttention + P2Detail | 1 | - | 0.492 | 0.000 | 0.492 | 0.492 | 0.817 |
| DySample | DySample | 1 | - | 0.491 | 0.000 | 0.491 | 0.491 | 0.827 |
| RT-DETR-R18 Baseline | No added module (baseline) | 4 | 0,1,-,- | 0.489 | 0.002 | 0.492 | 0.486 | 0.825 |
| SODGA-Calib | SODGuidedAttention + calibration | 2 | 0,1 | 0.489 | 0.008 | 0.497 | 0.482 | 0.815 |
| SODGA | SODGuidedAttention | 5 | 1,-,-,-,- | 0.489 | 0.009 | 0.507 | 0.481 | 0.830 |
| SODGA-Stable | SODGuidedAttentionStable | 1 | - | 0.485 | 0.000 | 0.485 | 0.485 | 0.827 |
| MSAInnov | MSAInnov | 1 | - | 0.484 | 0.000 | 0.484 | 0.484 | 0.824 |
| SODGA-CAFM | SODGuidedAttention + CAFM | 1 | - | 0.479 | 0.000 | 0.479 | 0.479 | 0.822 |
| IRS-PSConv | PSConv | 1 | 0 | 0.478 | 0.000 | 0.478 | 0.478 | 0.826 |
| MSFFFE | MSFFFE | 1 | - | 0.474 | 0.000 | 0.474 | 0.474 | 0.817 |

## Per-Run Detail

| Experiment/Config | Modules | Seed | Epochs | Best_P | Best_R | Best_mAP50 | Best_mAP50-95 | Final_mAP50-95 | Log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RT-DETR-R18 Baseline | No added module (baseline) | 0 | 150 | 0.868 | 0.825 | 0.830 | 0.486 | 0.486 | train_log/RT-DETR-R18/train_rtdetr_r18_Baseline-epoch150-Seed0.log |
| RT-DETR-R18 Baseline | No added module (baseline) | 1 | 150 | 0.912 | 0.800 | 0.810 | 0.491 | 0.491 | train_log/RT-DETR-R18/train_rtdetr_r18_Baseline-epoch150-Seed1.log |
| RT-DETR-R18 Baseline | No added module (baseline) | - | 100 | 0.884 | 0.814 | 0.831 | 0.489 | 0.489 | train_log/RT-DETR-R18/train_r18.log |
| RT-DETR-R18 Baseline | No added module (baseline) | - | 100 | 0.901 | 0.798 | 0.827 | 0.492 | 0.492 | train_log/RT-DETR-R18/train_rtdetr_r18_2.log |
| MSA-SODAttention-Original | MSAOriginalSODAttention | 0 | 150 | 0.897 | 0.818 | 0.839 | 0.501 | 0.501 | train_log/MSA-DETR/train_rtdetr_r18_MSA-SODAttention-Original-epoch150-Seed0.log |
| MSA-SODAttention-Original | MSAOriginalSODAttention | 1 | 150 | 0.898 | 0.823 | 0.844 | 0.494 | 0.494 | train_log/MSA-DETR/train_rtdetr_r18_MSA-SODAttention-Original-epoch150-Seed1.log |
| MSA-SODAttention-Original | MSAOriginalSODAttention | 2 | 150 | 0.886 | 0.820 | 0.831 | 0.497 | 0.497 | train_log/MSA-DETR/train_rtdetr_r18_MSA-SODAttention-Original-epoch150-Seed2.log |
| BCR beta=0.10 | BCRSODAttention(beta=0.10) | 0 | 150 | 0.870 | 0.818 | 0.830 | 0.499 | 0.499 | train_log/MSA-DETR/train_rtdetr_r18_MSA-SODAttention-BCR-epoch150-Seed0.log |
| BCR beta=0.10 | BCRSODAttention(beta=0.10) | 1 | 150 | 0.887 | 0.832 | 0.845 | 0.500 | 0.500 | train_log/MSA-DETR/train_rtdetr_r18_MSA-SODAttention-BCR-epoch150-Seed1.log |
| BCR beta=0.10 | BCRSODAttention(beta=0.10) | 2 | 150 | 0.897 | 0.808 | 0.827 | 0.501 | 0.501 | train_log/MSA-DETR/train_rtdetr_r18_MSA-SODAttention-BCR-epoch150-Seed2.log |
| BCR beta=0.05 | BCRSODAttention(beta=0.05) | 0 | 150 | 0.912 | 0.796 | 0.833 | 0.502 | 0.502 | train_log/MSA-DETR/train_rtdetr_r18_MSA-SODAttention-BCR-beta005-epoch150-Seed0.log |
| BCR beta=0.05 | BCRSODAttention(beta=0.05) | 1 | 150 | 0.925 | 0.787 | 0.816 | 0.488 | 0.488 | train_log/MSA-DETR/train_rtdetr_r18_MSA-SODAttention-BCR-beta005-epoch150-Seed1.log |
| BCR beta=0.05 | BCRSODAttention(beta=0.05) | 2 | 150 | 0.888 | 0.813 | 0.832 | 0.495 | 0.495 | train_log/MSA-DETR/train_rtdetr_r18_MSA-SODAttention-BCR-beta005-epoch150-Seed2.log |
| BCR beta=0.20 | BCRSODAttention(beta=0.20) | 0 | 150 | 0.881 | 0.830 | 0.847 | 0.492 | 0.492 | train_log/MSA-DETR/train_rtdetr_r18_MSA-SODAttention-BCR-beta020-epoch150-Seed0.log |
| BCR-DySample | BCRSODAttention + DySample | 0 | 150 | 0.906 | 0.816 | 0.842 | 0.500 | 0.500 | train_log/BCR-RTDETR/train_rtdetr_r18_BCR-DySample-epoch150-Seed0.log |
| BCR-DySample | BCRSODAttention + DySample | 1 | 150 | 0.921 | 0.777 | 0.818 | 0.490 | 0.489 | train_log/BCR-RTDETR/train_rtdetr_r18_BCR-DySample-epoch150-Seed1.log |
| BCR-DySample | BCRSODAttention + DySample | 2 | 150 | 0.902 | 0.799 | 0.826 | 0.502 | 0.502 | train_log/BCR-RTDETR/train_rtdetr_r18_BCR-DySample-epoch150-Seed2.log |
| BCR-SPDDownsample | BCRSODAttention + SPDDownsample | 0 | 150 | 0.916 | 0.811 | 0.813 | 0.493 | 0.493 | train_log/BCR-RTDETR/train_rtdetr_r18_BCR-SPDDownsample-epoch150-Seed0.log |
| SODGA | SODGuidedAttention | 1 | 100 | 0.888 | 0.816 | 0.819 | 0.481 | 0.480 | train_log/Other/train_rtdetr_r18_SODGA_seed1.log |
| SODGA | SODGuidedAttention | - | 100 | 0.894 | 0.813 | 0.831 | 0.485 | 0.485 | train_log/Other/train_rtdetr_r18_MSPConv-SODGA.log |
| SODGA | SODGuidedAttention | - | 100 | 0.893 | 0.835 | 0.843 | 0.507 | 0.507 | train_log/Other/train_rtdetr_r18_SODGA.log |
| SODGA | SODGuidedAttention | - | 100 | 0.903 | 0.812 | 0.833 | 0.491 | 0.491 | train_log/Other/train_rtdetr_r18_SODGA_2.log |
| SODGA | SODGuidedAttention | - | 100 | 0.901 | 0.805 | 0.822 | 0.483 | 0.483 | train_log/Other/train_rtdetr_r18_SODGA_3.log |
| SODGA-CAFM | SODGuidedAttention + CAFM | - | 100 | 0.887 | 0.814 | 0.822 | 0.479 | 0.479 | train_log/Other/train_rtdetr_r18_SODGA-CAFM.log |
| SODGA-Calib | SODGuidedAttention + calibration | 0 | 150 | 0.897 | 0.811 | 0.818 | 0.497 | 0.497 | train_log/MSA-DETR/train_rtdetr_r18_SODGA-Calib-epoch150-Seed0.log |
| SODGA-Calib | SODGuidedAttention + calibration | 1 | 150 | 0.890 | 0.792 | 0.813 | 0.482 | 0.482 | train_log/MSA-DETR/train_rtdetr_r18_SODGA-Calib-epoch150-Seed1.log |
| SODGA-P2Detail | SODGuidedAttention + P2Detail | - | 100 | 0.889 | 0.805 | 0.817 | 0.492 | 0.492 | train_log/Other/train_rtdetr_r18_SODGA-P2Detail.log |
| SODGA-Stable | SODGuidedAttentionStable | - | 100 | 0.915 | 0.803 | 0.827 | 0.485 | 0.485 | train_log/Other/train_rtdetr_r18_SODGA-Stable.log |
| DySample | DySample | - | 100 | 0.902 | 0.800 | 0.827 | 0.491 | 0.491 | train_log/Other/train_rtdetr_r18_DySample.log |
| H2Net | H2Net modules | 0 | 150 | 0.902 | 0.820 | 0.840 | 0.506 | 0.506 | train_log/H2Net/H2Net_epoch150-Seed0.log |
| H2Net | H2Net modules | - | 100 | 0.914 | 0.817 | 0.837 | 0.494 | 0.494 | train_log/H2Net/H2Net.log |
| H2Net | H2Net modules | - | 100 | 0.908 | 0.799 | 0.817 | 0.492 | 0.491 | train_log/H2Net/train_H2Net_2.log |
| HEMS-CSPMSEIE | CSPMSEIE / HEMS module | - | 100 | 0.891 | 0.801 | 0.822 | 0.497 | 0.497 | train_log/Other/train_rtdetr_r18_HEMS-CSPMSEIE.log |
| HFPG | HighFrequencyPriorGuidance | - | 100 | 0.898 | 0.803 | 0.822 | 0.496 | 0.496 | train_log/Other/train_rtdetr_r18_HFPG.log |
| IRS-PSConv | PSConv | 0 | 150 | 0.907 | 0.811 | 0.826 | 0.478 | 0.478 | train_log/IRS-DETR/train_rtdetr_r18_RT-DETR-R18-IRS-PSConv_epoch150-Seed0.log |
| MSAInnov | MSAInnov | - | 100 | 0.883 | 0.810 | 0.824 | 0.484 | 0.484 | train_log/MSA-DETR/train_rtdetr_r18_MSAInnov.log |
| MSAInnov-P2Decoder | MSAInnov + P2 decoder branch | - | 100 | 0.879 | 0.824 | 0.831 | 0.505 | 0.505 | train_log/MSA-DETR/train_rtdetr_r18_MSAInnov-P2Decoder.log |
| MSFFFE | MSFFFE | - | 100 | 0.914 | 0.780 | 0.817 | 0.474 | 0.474 | train_log/Other/train_rtdetr_r18_msfffe.log |
| MSPConv | MSPConv | - | 100 | 0.891 | 0.817 | 0.834 | 0.500 | 0.500 | train_log/Other/train_rtdetr_r18_MSPConv.log |
| P2Detail | P2Detail module | - | 100 | 0.915 | 0.815 | 0.844 | 0.498 | 0.498 | train_log/Other/train_rtdetr_r18_P2Detail.log |
