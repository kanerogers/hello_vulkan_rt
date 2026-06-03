struct Registers {
  uint textureID;
};

[[vk::push_constant]]
ConstantBuffer<Registers, ScalarDataLayout> registers;

[[vk::binding(0, 0)]]
Sampler2D<float4> textures[];

struct FragmentInput {
  float2 uv : TEXCOORD0;
}

[shader("fragment")]
float4 main(FragmentInput input) : SV_Target0 {
  let hdr = max(textures[NonUniformResourceIndex(registers.textureID)]
                    .Sample(input.uv)
                    .rgb,
                float3(0.0));

  let exposure = 0.35;
  let mapped = 1.0 - exp(-hdr * exposure);

  let gamma = 1.0 / 2.2;
  return float4(pow(saturate(mapped), float3(gamma)), 1.0);
}
