[[vk::push_constant]]
ConstantBuffer<Registers, ScalarDataLayout> registers;

[[vk::binding(0, 0)]]
RaytracingAccelerationStructure tlas;

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

struct HitPayload {
  float3 hitValue;
  uint seed;
  uint depth;
  float3 rayOrigin;
  float3 rayDirection;
  float3 weight;
};

struct ShadowPayload {
  bool hit;
}

struct TunnelLight {
  float3 position;
  float radiusMetres;
  float3 colour;
  float intensity;
  float startSMetres;
}

static const uint RAY_TYPE_RADIANCE = 0;
static const uint RAY_TYPE_SHADOW = 1;
static const uint RAY_TYPE_COUNT = 2;

static const uint MISS_TYPE_RADIANCE = 0;
static const uint MISS_TYPE_SHADOW = 1;
