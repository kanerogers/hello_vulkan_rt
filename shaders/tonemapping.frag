#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec2 outUV;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D textures[];

layout(push_constant) uniform Registers {
    uint textureID;
} registers;

void main()
{
    vec2 uv = outUV;
    float gamma = 1. / 2.2;
    fragColor = pow(texture(textures[nonuniformEXT(registers.textureID)], uv).rgba, vec4(gamma));
}
