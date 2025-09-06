/// Written lovingly by ChatGPT because I'm no good at this stuff
/// Common Vulkan-ish defaults. Prefer overriding at runtime withvalues you query.
pub const SCRATCH_ALIGN: u64 = 256; // minAccelerationStructureScratchOffsetAlignment (commonly 256)
pub const SBT_BASE_ALIGN: u64 = 64; // shaderGroupBaseAlignment (commonly 64)
pub const SBT_HANDLE_ALIGN: u64 = 32; // shaderGroupHandleAlignment (commonly 32)

#[inline]
pub const fn is_pow2(a: u64) -> bool {
    a != 0 && (a & (a - 1)) == 0
}

#[inline]
pub const fn is_aligned_pow2(value: u64, alignment: u64) -> bool {
    debug_assert!(is_pow2(alignment));
    (value & (alignment - 1)) == 0
}

#[inline]
pub const fn align_up_pow2(value: u64, alignment: u64) -> u64 {
    debug_assert!(is_pow2(alignment));
    (value + (alignment - 1)) & !(alignment - 1)
}

#[inline]
pub const fn align_down_pow2(value: u64, alignment: u64) -> u64 {
    debug_assert!(is_pow2(alignment));
    value & !(alignment - 1)
}

#[inline]
pub const fn padding_needed_pow2(value: u64, alignment: u64) -> u64 {
    debug_assert!(is_pow2(alignment));
    (alignment - (value & (alignment - 1))) & (alignment - 1)
}

/// Non-power-of-two versions (slower but safe).
#[inline]
pub const fn is_aligned_np2(value: u64, alignment: u64) -> bool {
    value % alignment == 0
}

#[inline]
pub const fn align_up_np2(value: u64, alignment: u64) -> u64 {
    let r = value % alignment;
    if r == 0 {
        value
    } else {
        value + (alignment - r)
    }
}

#[inline]
pub const fn align_down_np2(value: u64, alignment: u64) -> u64 {
    value - (value % alignment)
}

/// Integer ceil-div (e.g., tiles/dispatch sizing).
#[inline]
pub const fn ceil_div(n: u64, d: u64) -> u64 {
    (n + (d - 1)) / d
}

/// GCD/LCM for composing alignments (e.g., combine SBT base & handle align).
#[inline]
pub const fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % t;
        a = t;
    }
    a
}

#[inline]
pub const fn lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 {
        0
    } else {
        (a / gcd(a, b)) * b
    }
}

/// ---- Vulkan-adjacent helpers ----

/// Align an offset you plan to use as the AS build scratch address (suballocation within a big buffer).
#[inline]
pub const fn align_scratch_offset(offset: u64, scratch_align: u64) -> u64 {
    if is_pow2(scratch_align) {
        align_up_pow2(offset, scratch_align)
    } else {
        align_up_np2(offset, scratch_align)
    }
}

/// Compute an SBT record stride given the handle size, optional inline-data bytes,
/// and the required base/handle alignments. Stride must be:
///   - >= shaderGroupHandleSize + inline_bytes
///   - a multiple of shaderGroupBaseAlignment
/// (Some stacks also care about handle alignment; we satisfy both via LCM.)
#[inline]
pub const fn sbt_record_stride(
    shader_group_handle_size: u64,
    inline_bytes: u64,
    base_align: u64, // from VkPhysicalDeviceRayTracingPipelinePropertiesKHR.shaderGroupBaseAlignment
    handle_align: u64, // from ...shaderGroupHandleAlignment
) -> u64 {
    let need = shader_group_handle_size + inline_bytes;
    let align = if is_pow2(base_align) && is_pow2(handle_align) {
        lcm(base_align, handle_align)
    } else {
        // Fall back to base if non-pow2 weirdness; still valid.
        base_align
    };
    if is_pow2(align) {
        align_up_pow2(need, align)
    } else {
        align_up_np2(need, align)
    }
}

/// Size a whole SBT section (rgen/miss/hit), then align the *section end* to base alignment
/// (useful if you lay sections back-to-back in one buffer).
#[inline]
pub const fn sbt_section_size(count: u64, stride: u64, base_align: u64) -> u64 {
    let raw = count.saturating_mul(stride);
    if is_pow2(base_align) {
        align_up_pow2(raw, base_align)
    } else {
        align_up_np2(raw, base_align)
    }
}

/// Convenience: compute padding bytes to append to `current` so the next write meets `alignment`.
#[inline]
pub const fn pad_for(current: u64, alignment: u64) -> u64 {
    if is_pow2(alignment) {
        padding_needed_pow2(current, alignment)
    } else {
        (alignment - (current % alignment)) % alignment
    }
}

/// Example “typical Vulkan” wrappers using common defaults:
#[inline]
pub const fn align_scratch_offset_default(offset: u64) -> u64 {
    align_scratch_offset(offset, SCRATCH_ALIGN)
}

#[inline]
pub const fn sbt_record_stride_default(handle_size: u64, inline_bytes: u64) -> u64 {
    sbt_record_stride(handle_size, inline_bytes, SBT_BASE_ALIGN, SBT_HANDLE_ALIGN)
}

#[inline]
pub const fn sbt_section_size_default(count: u64, stride: u64) -> u64 {
    sbt_section_size(count, stride, SBT_BASE_ALIGN)
}
