struct Vertex {
  float3 position;
  float3 normal;
  float2 uv;
};

struct Material {
  float4 baseColourFactor;
  float3 emissiveColourFactor;
  uint baseColourTextureID;
  uint normalTextureID;
  uint metallicRoughnessTextureID;
  uint aoTextureID;
};

struct Primitive {
  Material *material;
  uint *indexBuffer;
  Vertex *vertexBuffer;
};

struct Registers {
  float4x4 viewInverse;
  float4x4 projInverse;
  Primitive *primitiveBuffer;
  float tunnelBayLengthMetres;
  uint frame;
  TunnelLight *lightBuffer;
  uint lightCount;
  float trainTrackSMetres;
  float trackLengthMetres;
};

[[vk::push_constant]]
ConstantBuffer<Registers, ScalarDataLayout> registers;

struct HitPayload {
  float3 hitValue;
  uint seed;
  uint depth;
  float3 rayOrigin;
  float3 rayDirection;
  float3 weight;
};

struct TunnelLight {
  float3 position;
  float radiusMetres;
  float3 colour;
  float intensity;
  float startSMetres;
}
