#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_ARB_shader_clock : enable

#include "common.glsl"
layout(location = 0) rayPayloadInEXT HitPayload payload;

void main() {
    if (payload.depth == 0) {
        payload.hitValue = vec3(0.1, 0.1, 0.2) * 0.8;
    } else {
        payload.hitValue = vec3(0.001); // tiny contribution from the environment
    }

    payload.depth = 100; // end the trace
}
