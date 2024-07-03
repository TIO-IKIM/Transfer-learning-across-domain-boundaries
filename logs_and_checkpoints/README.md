# README

All pretrained and finetuned models and their train/val/test logs live here. The names are still the naming scheme I used during the project, so they may seem a little confusing at first. All finetuning runs have the following structured name:

```
{Experiment Number}/PT_{Pretraining Method}_{Pretraining Dataset}_FT_{Finetuning Dataset}{Run Number}

```

PT and FT stand for pretraining and finetuning, respectively. The experiment number is one of \[E1, E2, E3, E4], where E1 is the regular finetuning experiment, E2 is the few-shot experiment, E3 is a noise experiment that never quite panned out and would take ages to complete, and E4 is the linear evaluation experiment. Run numbers are one of \["", "\_f1", "\_f2", "\_f3"].

Dataset aliases are:

```
BraTS => BraTS 2020
CX8   => ChestX-Ray8
I1k   => ImageNet-1k
LiTS  => LiTS 2017
R     => RadNet-1.28M
RF    => RadNet-12M (F means "Full" in this context)

```

Pretrained model weights live in:

```
No pretraining             => Scratch
Pretrained on ImageNet-1k  => sancheck_1k100
Pretrained on RadNet-1.28M => sancheck_ct100
Pretrained on RadNet-12M   => ct_full
Mixed, sequential          => IxR
Mixed, together            => IxR_mixed
```

If anything is missing, let me know. Occasionally, the upload chokes on a specific file and just ignores it and never tells you.


