"""Shared TorchVision ResNet-18 loading helper."""

from __future__ import annotations


def load_resnet18(weights_name: str):
    try:
        import torch
        import torchvision
        from torchvision.models import resnet18
    except ModuleNotFoundError as exc:
        missing = exc.name or "torch/torchvision"
        raise SystemExit(
            f"Missing dependency `{missing}`. Install torch and torchvision in "
            "the active Python environment before running this tool."
        ) from exc

    weights = None
    weights_label = "none"
    if weights_name != "none":
        try:
            from torchvision.models import ResNet18_Weights

            weights_map = {
                "default": ResNet18_Weights.DEFAULT,
                "imagenet1k_v1": ResNet18_Weights.IMAGENET1K_V1,
            }
            weights = weights_map[weights_name]
            weights_label = weights_name
        except (ImportError, AttributeError):
            if weights_name != "imagenet1k_v1":
                raise SystemExit(
                    "This torchvision version only supports the legacy "
                    "`pretrained=True` path for ResNet-18."
                )
            model = resnet18(pretrained=True)
            return torch, torchvision, model.eval(), "legacy_pretrained"

    model = resnet18(weights=weights)
    return torch, torchvision, model.eval(), weights_label

