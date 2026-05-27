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
  float gamma = 1.0 / 2.2;
  return pow(
      textures[NonUniformResourceIndex(registers.textureID)].Sample(input.uv),
      float4(gamma));
}
