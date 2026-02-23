import tqdm
from utils import send_inputs_to_device, evaluate_model, load_checkpoint, save_checkpoint, save_final_model
from dataset import split_data, split_context_data
from model import DTrOCRLMHeadModel
from config import DTrOCRConfig
from torch.utils.data import DataLoader
import torch
import multiprocessing as mp
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Train GraDeT-HTR with optional context-aware mode')
parser.add_argument('--context', action='store_true', help='Enable context-aware training')
parser.add_argument('--images_dir', type=str, default="../sample_train/images", help='Path to images directory')
parser.add_argument('--labels_file', type=str, default="../sample_train/labels/label.csv", help='Path to labels CSV (for non-context mode)')
parser.add_argument('--json_dir', type=str, default="../out/single_words/json", help='Path to JSON directory (for context mode)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--margin_lambda', type=float, default=0.5,
                    help='Lambda for margin loss term (context vs isolation)')
args = parser.parse_args()

config = DTrOCRConfig()

# Build dataset based on mode
if args.context:
    print("Using context-aware mode")
    train_dataset, validation_dataset = split_context_data(args.images_dir, args.json_dir, config, test_size=0.1)
else:
    print("Using standard mode (no context)")
    train_dataset, validation_dataset = split_data(args.images_dir, args.labels_file, config)

print(f"Train size: {len(train_dataset)}")
print(f"Test size: {len(validation_dataset)}")

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

# attempt to autodetect the device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print('using device: ', device)

# ensures reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

torch.set_float32_matmul_precision('high')

# Model
model = DTrOCRLMHeadModel(config)
#model = torch.compile(model)
model.to(device)

use_amp = True
scaler = torch.amp.GradScaler(device=device, enabled=use_amp)
optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)


# --- Bug 3 fix: evaluate_context_model defined BEFORE the training loop ---
def evaluate_context_model(model, dataloader, device, use_amp=True):
    """Evaluate model with context-aware dual-path loss."""
    model.eval()
    val_losses, val_ctx_losses, val_iso_losses = [], [], []
    val_accs = []

    with torch.no_grad():
        for inputs in dataloader:
            pixel_values = inputs['pixel_values'].to(device)
            labels = inputs['labels'].to(device)

            # Contextual
            input_ids_ctx = inputs['input_ids'].to(device)
            attention_mask_ctx = inputs['attention_mask'].to(device)
            context_length = inputs['context_length'][0].item() if isinstance(inputs['context_length'], torch.Tensor) else inputs['context_length']

            # Isolation
            input_ids_iso = inputs['input_ids_isolated'].to(device)
            attention_mask_iso = inputs['attention_mask_isolated'].to(device)
            labels_iso = inputs['labels_isolated'].to(device)
            context_length_iso = inputs['context_length_isolated'][0].item() if isinstance(inputs['context_length_isolated'], torch.Tensor) else inputs['context_length_isolated']

            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                outputs_ctx = model(pixel_values=pixel_values, input_ids=input_ids_ctx,
                                   attention_mask=attention_mask_ctx, labels=labels,
                                   context_length=context_length)
                outputs_iso = model(pixel_values=pixel_values, input_ids=input_ids_iso,
                                   attention_mask=attention_mask_iso, labels=labels_iso,
                                   context_length=context_length_iso)

            loss_net = outputs_ctx.loss + outputs_iso.loss
            val_losses.append(loss_net.item())
            val_ctx_losses.append(outputs_ctx.loss.item())
            val_iso_losses.append(outputs_iso.loss.item())
            val_accs.append((outputs_ctx.accuracy.item() + outputs_iso.accuracy.item()) / 2)

    return (sum(val_losses)/len(val_losses), sum(val_accs)/len(val_accs),
            sum(val_ctx_losses)/len(val_ctx_losses), sum(val_iso_losses)/len(val_iso_losses))


# Training
EPOCHS = args.epochs
train_losses, train_accuracies = [], []
validation_losses, validation_accuracies = [], []

if args.context:
    print("\n=== Training with Context-Aware Dual-Path Loss ===")
    print(f"L = -log(Pc) - log(Pi) + {args.margin_lambda}*max(0, loss_ctx - loss_iso)\n")

for epoch in range(EPOCHS):
    epoch_losses, epoch_accuracies = [], []
    epoch_ctx_losses, epoch_iso_losses = [], []
    epoch_ctx_accs, epoch_iso_accs = [], []
    epoch_margin_terms = []

    model.train()
    for inputs in tqdm.tqdm(train_dataloader, total=len(train_dataloader), desc=f'Epoch {epoch + 1}'):
        optimizer.zero_grad()

        # Move inputs to device
        pixel_values = inputs['pixel_values'].to(device)
        labels = inputs['labels'].to(device)

        if args.context:
            # === DUAL-PATH LOSS WITH MARGIN TERM ===

            # Contextual inputs
            input_ids_ctx = inputs['input_ids'].to(device)
            attention_mask_ctx = inputs['attention_mask'].to(device)
            context_length = inputs['context_length'][0].item() if isinstance(inputs['context_length'], torch.Tensor) else inputs['context_length']

            # Isolation inputs
            input_ids_iso = inputs['input_ids_isolated'].to(device)
            attention_mask_iso = inputs['attention_mask_isolated'].to(device)
            context_length_iso = inputs['context_length_isolated'][0].item() if isinstance(inputs['context_length_isolated'], torch.Tensor) else inputs['context_length_isolated']
            labels_iso = inputs['labels_isolated'].to(device)

            # Forward pass 1: Contextual mode
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                outputs_ctx = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids_ctx,
                    attention_mask=attention_mask_ctx,
                    labels=labels,
                    context_length=context_length,
                    return_per_sample_loss=True
                )
                loss_ctx = outputs_ctx.loss
                per_sample_loss_ctx = outputs_ctx.per_sample_loss
                acc_ctx = outputs_ctx.accuracy

            # Forward pass 2: Isolation mode
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                outputs_iso = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids_iso,
                    attention_mask=attention_mask_iso,
                    labels=labels_iso,
                    context_length=context_length_iso,
                    return_per_sample_loss=True
                )
                loss_iso = outputs_iso.loss
                per_sample_loss_iso = outputs_iso.per_sample_loss
                acc_iso = outputs_iso.accuracy

            # --- Bug 4 fix: Correct loss formula ---
            # L = -log(Pc) - log(Pi) + lambda * max(0, loss_ctx - loss_iso)
            # Terms 1+2: base cross-entropy from both paths
            combined_base_loss = per_sample_loss_ctx + per_sample_loss_iso

            # Term 3: margin penalty -- fires when context is WORSE than isolation
            margin_term = args.margin_lambda * torch.clamp(
                per_sample_loss_ctx - per_sample_loss_iso, min=0.0
            )

            weighted_loss = (combined_base_loss + margin_term).mean()

            # Backward pass
            scaler.scale(weighted_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Track metrics
            epoch_losses.append(weighted_loss.item())
            epoch_ctx_losses.append(loss_ctx.item())
            epoch_iso_losses.append(loss_iso.item())
            epoch_ctx_accs.append(acc_ctx.item())
            epoch_iso_accs.append(acc_iso.item())
            epoch_accuracies.append((acc_ctx.item() + acc_iso.item()) / 2)
            epoch_margin_terms.append(margin_term.mean().item())

        else:
            # === STANDARD TRAINING (NO CONTEXT) ===
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

            scaler.scale(outputs.loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(outputs.loss.item())
            epoch_accuracies.append(outputs.accuracy.item())

    # Store epoch metrics
    train_losses.append(sum(epoch_losses) / len(epoch_losses))
    train_accuracies.append(sum(epoch_accuracies) / len(epoch_accuracies))

    # Validation
    if args.context:
        val_loss, val_acc, val_ctx_loss, val_iso_loss = evaluate_context_model(model, validation_dataloader, device, use_amp)
        validation_losses.append(val_loss)
        validation_accuracies.append(val_acc)

        avg_margin = sum(epoch_margin_terms) / len(epoch_margin_terms) if epoch_margin_terms else 0.0
        print(f"\nEpoch {epoch + 1}/{EPOCHS}:")
        print(f"  Train - Loss: {train_losses[-1]:.4f} | Acc: {train_accuracies[-1]:.4f}")
        print(f"    Ctx Loss: {sum(epoch_ctx_losses)/len(epoch_ctx_losses):.4f} | Iso Loss: {sum(epoch_iso_losses)/len(epoch_iso_losses):.4f}")
        print(f"    Ctx Acc: {sum(epoch_ctx_accs)/len(epoch_ctx_accs):.4f} | Iso Acc: {sum(epoch_iso_accs)/len(epoch_iso_accs):.4f}")
        print(f"    Margin Mean: {avg_margin:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        print(f"    Ctx Loss: {val_ctx_loss:.4f} | Iso Loss: {val_iso_loss:.4f}")
    else:
        validation_loss, validation_accuracy = evaluate_model(model, validation_dataloader, device=device)
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)

        print(f"Epoch: {epoch + 1} - Train loss: {train_losses[-1]:.4f}, Train accuracy: {train_accuracies[-1]:.4f}, Validation loss: {validation_losses[-1]:.4f}, Validation accuracy: {validation_accuracies[-1]:.4f}")

    # --- Bug 6 fix: save_checkpoint with correct arguments ---
    save_checkpoint(
        model, optimizer, epoch + 1,
        train_losses[-1], validation_losses[-1],
        train_accuracies[-1], validation_accuracies[-1],
        './',
        f'checkpoint_{"context_" if args.context else ""}epoch_{epoch+1}.pt'
    )

# Save final model
save_final_model(model, f'final_{"context_" if args.context else ""}model.pth')
print("\nTraining complete!")
