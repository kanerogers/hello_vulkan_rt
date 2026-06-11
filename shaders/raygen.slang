import common;
import sampling;

[[vk::binding(1, 0)]]
[vk::image_format("rgba16f")]
RWTexture2D<float4> image;

static const uint CAMERA_SAMPLE_COUNT = 4;

float2 cameraSampleOffset(uint sampleIndex) {
  if (sampleIndex == 0)
    return float2(0.375, 0.125);
  if (sampleIndex == 1)
    return float2(0.875, 0.375);
  if (sampleIndex == 2)
    return float2(0.125, 0.625);
  return float2(0.625, 0.875);
}

float3 traceCameraSample(float2 pixelPosition, uint3 launchSize, uint seed) {
  float2 inUV = pixelPosition / float2(launchSize.xy);
  float2 d = inUV * 2.0 - 1.0;

  float4 origin = mul(registers.viewInverse, float4(0, 0, 0, 1));
  float4 target = mul(registers.projInverse, float4(d.x, d.y, 1, 1));
  float4 direction =
      mul(registers.viewInverse, float4(normalize(target.xyz), 0));

  HitPayload payload;
  payload.hitValue = float3(0);
  payload.seed = seed;
  payload.depth = 0;
  payload.rayOrigin = origin.xyz;
  payload.rayDirection = direction.xyz;
  payload.weight = float3(0);

  float3 curWeight = float3(1);
  float3 hitValue = float3(0);

  for (; payload.depth < 10; payload.depth++) {
    RayDesc ray;
    ray.Origin = payload.rayOrigin;
    ray.TMin = 0.001;
    ray.Direction = payload.rayDirection;
    ray.TMax = 1000.0;

    TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, RAY_TYPE_RADIANCE,
             HIT_SBT_STRIDE_IGNORE_GEOMETRY, MISS_TYPE_RADIANCE, ray, payload);

    hitValue += payload.hitValue * curWeight;
    curWeight *= payload.weight;
  }

  return hitValue;
}

[shader("raygeneration")]
void main() {
  uint3 launchID = DispatchRaysIndex();
  uint3 launchSize = DispatchRaysDimensions();
  int2 imageCoords = int2(launchID.xy);

  float3 colour = float3(0.0);

  for (uint sampleIndex = 0; sampleIndex < CAMERA_SAMPLE_COUNT; sampleIndex++) {
    let pixelPosition = float2(launchID.xy) + cameraSampleOffset(sampleIndex);

    let seed =
        tea(launchID.y * launchSize.x + launchID.x,
            launchID.x * 1973u + launchID.y * 9277u + sampleIndex * 26699u);

    colour += traceCameraSample(pixelPosition, launchSize, seed);
  }

  colour /= float(CAMERA_SAMPLE_COUNT);
  image[imageCoords] = float4(colour, 1);
}
