from pathlib import Path

import loralib as lora
import torch


def expected_lora_keys(model):
    return {name for name, _ in model.named_parameters() if "lora_" in name}


def load_lora_state(model, state):
    expected = expected_lora_keys(model)
    provided = {key for key in state if "lora_" in key}
    missing = sorted(expected - provided)
    unexpected = sorted(provided - expected)
    if missing or unexpected:
        raise RuntimeError(
            "LoRA checkpoint is incompatible with the model. "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    model.load_state_dict(state, strict=False)


def save_stage1_checkpoint(
    path,
    model,
    ssl_head,
    optimizer,
    scheduler,
    scaler,
    epoch,
    global_step,
    config,
    metrics,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "stage": "stage1_ssl",
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
        "metrics": metrics,
        "lora_state": lora.lora_state_dict(model, bias="none"),
        "ssl_head_state": ssl_head.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_stage2_checkpoint(
    path,
    model,
    classifier,
    contrastive_head,
    ema_model,
    ema_classifier,
    optimizer,
    scheduler,
    scaler,
    model_probability,
    epoch,
    global_step,
    config,
    metrics,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "stage": "stage2_semi",
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
        "metrics": metrics,
        "lora_state": lora.lora_state_dict(model, bias="none"),
        "classifier_state": classifier.state_dict(),
        "contrastive_head_state": contrastive_head.state_dict(),
        "ema_lora_state": lora.lora_state_dict(ema_model, bias="none"),
        "ema_classifier_state": ema_classifier.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "model_probability": model_probability.detach().cpu(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
