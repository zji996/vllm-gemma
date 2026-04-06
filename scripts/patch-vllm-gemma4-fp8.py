#!/usr/bin/env python3
"""
Patch vLLM gemma4.py so Gemma4 MoE expert FP8 scales can be loaded.

The stock v0.19.0 Gemma4 loader maps expert weights, but not expert
`weight_scale` tensors. Our offline FP8 checkpoint writes per-expert:

  - experts.{i}.gate_proj
  - experts.{i}.up_proj
  - experts.{i}.down_proj
  - experts.{i}.gate_proj.weight_scale
  - experts.{i}.up_proj.weight_scale
  - experts.{i}.down_proj.weight_scale

This patch teaches the loader to route those scales into:

  - experts.w13_weight_scale
  - experts.w2_weight_scale
"""

from pathlib import Path
import sys


VLLM_ROOT_CANDIDATES = [
    Path("/opt/vllm-src/vllm"),
    Path("/usr/local/lib/python3.12/dist-packages/vllm"),
]


def resolve_vllm_root() -> Path:
    for root in VLLM_ROOT_CANDIDATES:
        if (root / "model_executor/models/gemma4.py").exists() and (
            root / "_custom_ops.py"
        ).exists():
            return root
    raise FileNotFoundError(
        "Could not find vLLM install root. Checked: "
        + ", ".join(str(p) for p in VLLM_ROOT_CANDIDATES)
    )


OLD_REMAP = """                if ".experts.gate_up_proj" in name:
                    name = name.replace(
                        ".experts.gate_up_proj",
                        ".moe.gate_up_proj",
                    )
                elif ".experts.down_proj" in name:
                    name = name.replace(
                        ".experts.down_proj",
                        ".moe.down_proj",
                    )
"""

NEW_REMAP = """                if ".experts.gate_up_proj" in name:
                    name = name.replace(
                        ".experts.gate_up_proj",
                        ".moe.gate_up_proj",
                    )
                elif ".experts.down_proj" in name:
                    name = name.replace(
                        ".experts.down_proj",
                        ".moe.down_proj",
                    )

                # Support offline-expanded per-expert Gemma4 MoE FP8 checkpoints:
                #   layers.N.experts.{id}.{gate,up,down}_proj(.weight_scale)
                # Remap them to the internal vLLM naming that includes `.moe.`
                name = re.sub(
                    r"(layers\\.\\d+)\\.experts\\.(\\d+)\\.",
                    r"\\1.moe.experts.\\2.",
                    name,
                )
"""


OLD_MAPPING = """        expert_params_mapping = [
            # (param_name, weight_name, expert_id, shard_id)
            (
                "experts.w13_weight"
                if proj_name in ["gate_proj", "up_proj"]
                else "experts.w2_weight",
                f"experts.{expert_id}.{proj_name}",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_experts)
            for shard_id, proj_name in [
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ]
        ]
"""

NEW_MAPPING = """        expert_params_mapping = [
            # (param_name, weight_name, expert_id, shard_id)
            (
                "experts.w13_weight"
                if proj_name in ["gate_proj", "up_proj"]
                else "experts.w2_weight",
                f"experts.{expert_id}.{proj_name}",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_experts)
            for shard_id, proj_name in [
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ]
        ]
        expert_scale_mapping = [
            # (param_name, weight_name, expert_id, shard_id)
            (
                "experts.w13_weight_scale"
                if proj_name in ["gate_proj", "up_proj"]
                else "experts.w2_weight_scale",
                f"experts.{expert_id}.{proj_name}.weight_scale",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_experts)
            for shard_id, proj_name in [
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ]
        ]
"""


OLD_LOOP = """                for (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    moe_name = name.replace(weight_name, param_name)
                    if moe_name not in params_dict:
                        continue
                    if is_pp_missing_parameter(moe_name, self):
                        continue
                    param = params_dict[moe_name]
                    # Expert weights are already in the correct
                    # orientation for FusedMoE after _weight_iterator:
                    #   gate/up: [I, H] → w1/w3 expects [I, H]
                    #   down:    [H, I] → w2 expects [H, I]
                    assert loaded_weight.dim() == 2, (
                        f"Expected 2D expert weight for {weight_name}, "
                        f"got shape {loaded_weight.shape}"
                    )
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        weight_name + ".weight",
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    loaded_params.add(moe_name)
                    break
                else:
"""

NEW_LOOP = """                for (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in expert_scale_mapping:
                    if weight_name not in name:
                        continue
                    moe_name = name.replace(weight_name, param_name)
                    if moe_name not in params_dict:
                        continue
                    if is_pp_missing_parameter(moe_name, self):
                        continue
                    param = params_dict[moe_name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        weight_name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    loaded_params.add(moe_name)
                    break
                else:
                    for (
                        param_name,
                        weight_name,
                        expert_id,
                        shard_id,
                    ) in expert_params_mapping:
                        if weight_name not in name:
                            continue
                        moe_name = name.replace(weight_name, param_name)
                        if moe_name not in params_dict:
                            continue
                        if is_pp_missing_parameter(moe_name, self):
                            continue
                        param = params_dict[moe_name]
                        # Expert weights are already in the correct
                        # orientation for FusedMoE after _weight_iterator:
                        #   gate/up: [I, H] → w1/w3 expects [I, H]
                        #   down:    [H, I] → w2 expects [H, I]
                        assert loaded_weight.dim() == 2, (
                            f"Expected 2D expert weight for {weight_name}, "
                            f"got shape {loaded_weight.shape}"
                        )
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            weight_name + ".weight",
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                        loaded_params.add(moe_name)
                        break
                    else:
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        name = maybe_remap_kv_scale_name(name, params_dict)
                        if name is None:
                            continue
                        if is_pp_missing_parameter(name, self):
                            continue
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
"""

EXPERT_DIRECT_BLOCK = """            direct_expert_match = re.match(
                r"^(layers\\.\\d+\\.moe\\.)experts\\.(\\d+)\\."
                r"(gate_proj|up_proj|down_proj)(\\.weight_scale)?$",
                name,
            )
            if direct_expert_match is not None:
                layer_prefix, expert_id_str, proj_name, scale_suffix = (
                    direct_expert_match.groups()
                )
                expert_id = int(expert_id_str)
                shard_id = {
                    "gate_proj": "w1",
                    "down_proj": "w2",
                    "up_proj": "w3",
                }[proj_name]
                is_scale = scale_suffix is not None
                if proj_name in ("gate_proj", "up_proj"):
                    param_name = (
                        "experts.w13_weight_scale"
                        if is_scale
                        else "experts.w13_weight"
                    )
                else:
                    param_name = (
                        "experts.w2_weight_scale"
                        if is_scale
                        else "experts.w2_weight"
                    )
                moe_name = f"{layer_prefix}{param_name}"
                if moe_name in params_dict and not is_pp_missing_parameter(
                    moe_name, self
                ):
                    param = params_dict[moe_name]
                    weight_loader = param.weight_loader
                    if is_scale:
                        weight_loader(
                            param,
                            loaded_weight,
                            f"experts.{expert_id}.{proj_name}.weight_scale",
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    else:
                        assert loaded_weight.dim() == 2, (
                            f"Expected 2D expert weight for {proj_name}, "
                            f"got shape {loaded_weight.shape}"
                        )
                        weight_loader(
                            param,
                            loaded_weight,
                            f"experts.{expert_id}.{proj_name}.weight",
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    loaded_params.add(moe_name)
                    continue
"""

OLD_SCALED_FP8_QUANT = """def scaled_fp8_quant(
    input: torch.Tensor,
    scale: torch.Tensor | None = None,
    num_token_padding: int | None = None,
    scale_ub: torch.Tensor | None = None,
    use_per_token_if_dynamic: bool = False,
    output: torch.Tensor | None = None,
    group_shape: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    \"\"\"
    Quantize input tensor to FP8 and return quantized tensor and scale.

    This function supports both static and dynamic quantization: If you
    provide the scale, it will use static scaling and if you omit it,
    the scale will be determined dynamically. The function also allows
    optional padding of the output tensors for downstream kernels that
    will benefit from padding.

    Args:
        input: The input tensor to be quantized to FP8 (must be 2D: [M, N])
        scale: Optional scaling factor for the FP8 quantization. Supports:
            - 0D or [1]: per-tensor scaling
            - 1D: requires explicit group_shape to disambiguate per-channel
              vs per-token (use (-1, 1) for per-channel, (1, -1) for per-token)
            - 2D [M/group_m, N/group_n]: group scaling (e.g. [M, N/128] for
              DeepSeek-style (1,128) groups, or [M/128, N/128] for (128,128))
        scale_ub: Optional upper bound for scaling factor in dynamic
            per token case
        num_token_padding: If specified, pad the first dimension
            of the output to at least this value.
        use_per_token_if_dynamic: Whether to do per_tensor or per_token
            in the dynamic quantization case.
        group_shape: Optional tuple (group_m, group_n) specifying the group
            shape for static quantization. Use -1 for \"full extent\" (e.g.,
            (-1, -1) for per-tensor, (-1, 1) for per-channel, etc.)
            Required for 1D scales; optional for 2D scales.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
            scaling factor.
    \"\"\"
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    shape: tuple[int, int] | torch.Size = input.shape
    # For ROCm on MI300, the output fp8 dtype is torch.float_e3m3fnuz
    out_dtype: torch.dtype = current_platform.fp8_dtype()
    if num_token_padding:
        shape = (max(num_token_padding, input.shape[0]), shape[1])
    if output is None:
        output = torch.empty(shape, device=input.device, dtype=out_dtype)
    else:
        assert num_token_padding is None, \"padding not supported if output passed in\"
        assert output.dtype == out_dtype

    if scale is None:
        if use_per_token_if_dynamic:
            scale = torch.empty((shape[0], 1), device=input.device, dtype=torch.float32)
            torch.ops._C.dynamic_per_token_scaled_fp8_quant(
                output, input, scale, scale_ub
            )
        else:
            scale = torch.empty(1, device=input.device, dtype=torch.float32)
            torch.ops._C.dynamic_scaled_fp8_quant(output, input, scale)
    else:
        torch.ops._C.static_scaled_fp8_quant(output, input, scale, group_shape)

    return output, scale
"""

NEW_SCALED_FP8_QUANT = """def scaled_fp8_quant(
    input: torch.Tensor,
    scale: torch.Tensor | None = None,
    num_token_padding: int | None = None,
    scale_ub: torch.Tensor | None = None,
    use_per_token_if_dynamic: bool = False,
    output: torch.Tensor | None = None,
    group_shape: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    \"\"\"
    Quantize input tensor to FP8 and return quantized tensor and scale.

    This function supports both static and dynamic quantization: If you
    provide the scale, it will use static scaling and if you omit it,
    the scale will be determined dynamically. The function also allows
    optional padding of the output tensors for downstream kernels that
    will benefit from padding.

    Args:
        input: The input tensor to be quantized to FP8 (must be 2D: [M, N])
        scale: Optional scaling factor for the FP8 quantization. Supports:
            - 0D or [1]: per-tensor scaling
            - 1D: requires explicit group_shape to disambiguate per-channel
              vs per-token (use (-1, 1) for per-channel, (1, -1) for per-token)
            - 2D [M/group_m, N/group_n]: group scaling (e.g. [M, N/128] for
              DeepSeek-style (1,128) groups, or [M/128, N/128] for (128,128))
        scale_ub: Optional upper bound for scaling factor in dynamic
            per token case
        num_token_padding: If specified, pad the first dimension
            of the output to at least this value.
        use_per_token_if_dynamic: Whether to do per_tensor or per_token
            in the dynamic quantization case.
        group_shape: Optional tuple (group_m, group_n) specifying the group
            shape for static quantization. Use -1 for \"full extent\" (e.g.,
            (-1, -1) for per-tensor, (-1, 1) for per-channel, etc.)
            Required for 1D scales; optional for 2D scales.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
            scaling factor.
    \"\"\"
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    shape: tuple[int, int] | torch.Size = input.shape
    # For ROCm on MI300, the output fp8 dtype is torch.float_e3m3fnuz
    out_dtype: torch.dtype = current_platform.fp8_dtype()
    if num_token_padding:
        shape = (max(num_token_padding, input.shape[0]), shape[1])
    if output is None:
        output = torch.empty(shape, device=input.device, dtype=out_dtype)
    else:
        assert num_token_padding is None, \"padding not supported if output passed in\"
        assert output.dtype == out_dtype

    if scale is None:
        if use_per_token_if_dynamic:
            scale = torch.empty((shape[0], 1), device=input.device, dtype=torch.float32)
            torch.ops._C.dynamic_per_token_scaled_fp8_quant(
                output, input, scale, scale_ub
            )
        else:
            scale = torch.empty(1, device=input.device, dtype=torch.float32)
            torch.ops._C.dynamic_scaled_fp8_quant(output, input, scale)
        return output, scale

    capability = current_platform.get_device_capability()
    capability_int = -1 if capability is None else capability.to_int()

    # Prebuilt vLLM wheels can miss the static FP8 quant kernel image for
    # Ampere/SM86. Fall back to pure torch static quantization during weight
    # loading so offline FP8 checkpoints still work on 2x RTX 3080.
    if capability_int < 89:
        fp8_min = -224.0 if current_platform.is_fp8_fnuz() else torch.finfo(out_dtype).min
        fp8_max = 224.0 if current_platform.is_fp8_fnuz() else torch.finfo(out_dtype).max
        output.copy_(torch.clamp(input / scale, min=fp8_min, max=fp8_max).to(out_dtype))
        return output, scale

    try:
        torch.ops._C.static_scaled_fp8_quant(output, input, scale, group_shape)
    except torch.AcceleratorError as exc:
        if \"no kernel image is available for execution on the device\" not in str(exc):
            raise
        logger.warning(
            \"Falling back to torch static FP8 quantization because the \"
            \"prebuilt vLLM kernel does not support this GPU architecture.\"
        )
        fp8_min = -224.0 if current_platform.is_fp8_fnuz() else torch.finfo(out_dtype).min
        fp8_max = 224.0 if current_platform.is_fp8_fnuz() else torch.finfo(out_dtype).max
        output.copy_(torch.clamp(input / scale, min=fp8_min, max=fp8_max).to(out_dtype))

    return output, scale
"""

OLD_PP_PER_LAYER_INPUTS = """            per_layer_inputs = intermediate_tensors.get("per_layer_inputs")
"""

NEW_PP_PER_LAYER_INPUTS = """            per_layer_inputs = intermediate_tensors.tensors.get("per_layer_inputs")
"""

OLD_PP_RETURN_INTERMEDIATE = """        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                    "per_layer_inputs": per_layer_inputs,
                }
            )
"""

PREV_PP_RETURN_INTERMEDIATE = """        if not get_pp_group().is_last_rank:
            intermediate_tensors = {
                "hidden_states": hidden_states,
                "residual": residual,
            }
            if per_layer_inputs is not None:
                intermediate_tensors["per_layer_inputs"] = per_layer_inputs
            return IntermediateTensors(intermediate_tensors)
"""

PREV_PP_RETURN_INTERMEDIATE_WITH_PLI_ZERO = """        if not get_pp_group().is_last_rank:
            intermediate_tensors = {
                "hidden_states": hidden_states,
                "residual": residual,
            }
            if self.hidden_size_per_layer_input:
                if per_layer_inputs is None:
                    per_layer_inputs = torch.zeros(
                        (
                            hidden_states.shape[0],
                            self.config.num_hidden_layers,
                            self.hidden_size_per_layer_input,
                        ),
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    )
                intermediate_tensors["per_layer_inputs"] = per_layer_inputs
            return IntermediateTensors(intermediate_tensors)
"""

NEW_PP_RETURN_INTERMEDIATE = """        if not get_pp_group().is_last_rank:
            if residual is None:
                residual = torch.zeros_like(hidden_states)
            intermediate_tensors = {
                "hidden_states": hidden_states,
                "residual": residual,
            }
            if self.hidden_size_per_layer_input:
                if per_layer_inputs is None:
                    per_layer_inputs = torch.zeros(
                        (
                            hidden_states.shape[0],
                            self.config.num_hidden_layers,
                            self.hidden_size_per_layer_input,
                        ),
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    )
                intermediate_tensors["per_layer_inputs"] = per_layer_inputs
            return IntermediateTensors(intermediate_tensors)
"""


def main() -> int:
    vllm_root = resolve_vllm_root()
    target = vllm_root / "model_executor/models/gemma4.py"
    custom_ops_target = vllm_root / "_custom_ops.py"

    text = target.read_text()
    if "expert_scale_mapping" in text:
        # still allow updating the remap block if this script evolves
        pass

    if OLD_REMAP in text:
        text = text.replace(OLD_REMAP, NEW_REMAP, 1)
    elif "layers\\.\\d+)\\.experts\\.(\\d+)" not in text:
        print("Failed to find expected gemma4.py remap anchor", file=sys.stderr)
        return 1

    if "expert_scale_mapping" not in text:
        if OLD_MAPPING not in text or OLD_LOOP not in text:
            print("Failed to find expected gemma4.py anchors", file=sys.stderr)
            return 1
        text = text.replace(OLD_MAPPING, NEW_MAPPING, 1)
        text = text.replace(OLD_LOOP, NEW_LOOP, 1)

    direct_anchor = """            for param_name, shard_name, shard_id in stacked_params_mapping:
"""
    if "direct_expert_match = re.match(" not in text:
        if direct_anchor not in text:
            print("Failed to find direct expert insertion anchor", file=sys.stderr)
            return 1
        text = text.replace(direct_anchor, EXPERT_DIRECT_BLOCK + "\n" + direct_anchor, 1)

    if OLD_PP_PER_LAYER_INPUTS in text:
        text = text.replace(OLD_PP_PER_LAYER_INPUTS, NEW_PP_PER_LAYER_INPUTS, 1)
    elif NEW_PP_PER_LAYER_INPUTS not in text:
        print("Failed to find expected gemma4.py PP forward anchor", file=sys.stderr)
        return 1

    if OLD_PP_RETURN_INTERMEDIATE in text:
        text = text.replace(OLD_PP_RETURN_INTERMEDIATE, NEW_PP_RETURN_INTERMEDIATE, 1)
    elif PREV_PP_RETURN_INTERMEDIATE in text:
        text = text.replace(
            PREV_PP_RETURN_INTERMEDIATE,
            NEW_PP_RETURN_INTERMEDIATE,
            1,
        )
    elif PREV_PP_RETURN_INTERMEDIATE_WITH_PLI_ZERO in text:
        text = text.replace(
            PREV_PP_RETURN_INTERMEDIATE_WITH_PLI_ZERO,
            NEW_PP_RETURN_INTERMEDIATE,
            1,
        )
    elif NEW_PP_RETURN_INTERMEDIATE not in text:
        print("Failed to find expected gemma4.py PP return anchor", file=sys.stderr)
        return 1

    target.write_text(text)
    print(f"Patched {target}")

    custom_ops_text = custom_ops_target.read_text()
    if "prebuilt vLLM kernel does not support this GPU architecture" not in custom_ops_text:
        if OLD_SCALED_FP8_QUANT not in custom_ops_text:
            print("Failed to find expected _custom_ops.py anchor", file=sys.stderr)
            return 1
        custom_ops_text = custom_ops_text.replace(
            OLD_SCALED_FP8_QUANT,
            NEW_SCALED_FP8_QUANT,
            1,
        )
        custom_ops_target.write_text(custom_ops_text)
        print(f"Patched {custom_ops_target}")
    else:
        print(f"Already patched {custom_ops_target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
