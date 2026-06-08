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
static const uint RAY_TYPE_COUNT = 1;

static const uint MISS_TYPE_RADIANCE = 0;
static const uint MISS_TYPE_SHADOW = 1;

static const uint BAY_GEOMETRY_KIND_COUNT = 7;
static const uint GEOMETRY_KIND_TUNNEL_SHELL = 0;
static const uint HIT_SBT_STRIDE_IGNORE_GEOMETRY = 0;

static const uint GEOMETRY_KIND_SLAB_BED = 1;
static const uint GEOMETRY_KIND_RAILS = 2;
static const uint GEOMETRY_KIND_SERVICE_WALKWAY = 3;
static const uint GEOMETRY_KIND_CABLE_TRAY = 4;
static const uint GEOMETRY_KIND_LED_TUBES = 5;
static const uint GEOMETRY_KIND_LOWER_STRIPS = 6;

static const uint LOWER_STRIP_LIGHTS_PER_BAY = 25;
static const uint TUNNEL_LIGHTS_PER_BAY = 1 + LOWER_STRIP_LIGHTS_PER_BAY;
static const uint LED_LIGHT_OFFSET_IN_BAY = 0;
static const uint LOWER_STRIP_LIGHT_OFFSET_IN_BAY = 1;
