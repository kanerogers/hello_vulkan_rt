struct Registers {
    uint textureID;
};

[[vk::push_constant]]
ConstantBuffer<Registers, ScalarDataLayout> registers;

[[vk::binding(0, 0)]]
Sampler2D<float4> textures[];

[shader("fragment")]
float4 main([[vk::location(0)]] float2 uv: TEXCOORD0) : SV_Target0
{
    float gamma = 1.0 / 2.2;
    return pow(textures[NonUniformResourceIndex(registers.textureID)].Sample(uv), float4(gamma));
}

