============================================================
CONTEXT BENEFIT ANALYSIS
============================================================

Final Validation Metrics:
  Context Loss:   0.2668
  Isolation Loss: 0.2751
  Loss Improvement: 0.0082 (3.0%)

  Context Accuracy:   0.9202
  Isolation Accuracy: 0.9195
  Accuracy Improvement: 0.0007

Context HELPS: Model performs better with previous word context.

Epoch-by-epoch context benefit (ctx_loss - iso_loss):
  Epoch  1: -0.0186 
  Epoch  2: -0.0162 
  Epoch  3: -0.0140 
  Epoch  4: -0.0092 
  Epoch  5: -0.0121 
  Epoch  6: -0.0087 
  Epoch  7: -0.0082 



Starting Stage 3 training: 7 epochs
Dual-path loss: L = ctx_loss + iso_loss + 0.7 * max(0, ctx_loss - iso_loss)
======================================================================
Epoch 1/7: 100%|██████████| 1440/1440 [27:13<00:00,  1.13s/it, loss=0.8845, ctx=0.4099, iso=0.4746]
Validating: 100%|██████████| 76/76 [00:29<00:00,  2.61it/s]

Epoch 1/7
  Train: loss=2.8196, ctx=1.4013, iso=1.4066
  Val:   ctx_loss=0.6507, iso_loss=0.6693
         ctx_acc=0.8112,  iso_acc=0.8067
  Saved: stage3_checkpoint_epoch_1.pt (local + GDrive)
Epoch 2/7: 100%|██████████| 1440/1440 [27:14<00:00,  1.13s/it, loss=0.7756, ctx=0.3592, iso=0.4164]
Validating: 100%|██████████| 76/76 [00:29<00:00,  2.61it/s]

Epoch 2/7
  Train: loss=0.8818, ctx=0.4310, iso=0.4459
  Val:   ctx_loss=0.4342, iso_loss=0.4504
         ctx_acc=0.8731,  iso_acc=0.8682
  Saved: stage3_checkpoint_epoch_2.pt (local + GDrive)
Epoch 3/7: 100%|██████████| 1440/1440 [27:14<00:00,  1.14s/it, loss=0.6055, ctx=0.3094, iso=0.2650]
Validating: 100%|██████████| 76/76 [00:29<00:00,  2.61it/s]

Epoch 3/7
  Train: loss=0.5736, ctx=0.2790, iso=0.2905
  Val:   ctx_loss=0.3736, iso_loss=0.3876
         ctx_acc=0.8919,  iso_acc=0.8874
  Saved: stage3_checkpoint_epoch_3.pt (local + GDrive)
Epoch 4/7: 100%|██████████| 1440/1440 [27:14<00:00,  1.14s/it, loss=0.4052, ctx=0.1981, iso=0.2071]
Validating: 100%|██████████| 76/76 [00:29<00:00,  2.61it/s]

Epoch 4/7
  Train: loss=0.4345, ctx=0.2116, iso=0.2190
  Val:   ctx_loss=0.3147, iso_loss=0.3239
         ctx_acc=0.9066,  iso_acc=0.9029
  Saved: stage3_checkpoint_epoch_4.pt (local + GDrive)
Epoch 5/7: 100%|██████████| 1440/1440 [27:14<00:00,  1.13s/it, loss=0.4638, ctx=0.2268, iso=0.2370]
Validating: 100%|██████████| 76/76 [00:29<00:00,  2.61it/s]

Epoch 5/7
  Train: loss=0.3489, ctx=0.1693, iso=0.1764
  Val:   ctx_loss=0.3036, iso_loss=0.3158
         ctx_acc=0.9112,  iso_acc=0.9075
  Saved: stage3_checkpoint_epoch_5.pt (local + GDrive)
Epoch 6/7: 100%|██████████| 1440/1440 [27:14<00:00,  1.13s/it, loss=0.1852, ctx=0.0865, iso=0.0987]
Validating: 100%|██████████| 76/76 [00:29<00:00,  2.61it/s]

Epoch 6/7
  Train: loss=0.2892, ctx=0.1398, iso=0.1467
  Val:   ctx_loss=0.2759, iso_loss=0.2846
         ctx_acc=0.9180,  iso_acc=0.9151
  Saved: stage3_checkpoint_epoch_6.pt (local + GDrive)
Epoch 7/7: 100%|██████████| 1440/1440 [27:14<00:00,  1.13s/it, loss=0.2077, ctx=0.0956, iso=0.1121]
Validating: 100%|██████████| 76/76 [00:29<00:00,  2.61it/s]

Epoch 7/7
  Train: loss=0.2452, ctx=0.1190, iso=0.1233
  Val:   ctx_loss=0.2668, iso_loss=0.2751
         ctx_acc=0.9202,  iso_acc=0.9195
  Saved: stage3_checkpoint_epoch_7.pt (local + GDrive)

======================================================================