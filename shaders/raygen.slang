import common;
import sampling;

[[vk::binding(1, 0)]]
[vk::image_format("rgba16f")]
RWTexture2D<float4> image;

[shader("raygeneration")]
void main() {
  uint3 launchID = DispatchRaysIndex();
  uint3 launchSize = DispatchRaysDimensions();
  float2 pixelCentre = float2(launchID.xy) + float2(0.5);
  float2 inUV = pixelCentre / float2(launchSize.xy);
  int2 imageCoords = int2(launchID.xy);
  float2 d = inUV * 2.0 - 1.0;

  float4 origin = mul(registers.viewInverse, float4(0, 0, 0, 1));
  float4 target = mul(registers.projInverse, float4(d.x, d.y, 1, 1));
  float4 direction =
      mul(registers.viewInverse, float4(normalize(target.xyz), 0));

  uint seed = tea(launchID.y * launchSize.x + launchID.x,
                  launchID.x * 1973u + launchID.y * 9277u + registers.frame);

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
    ray.TMax = 10000.0;

    TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, RAY_TYPE_RADIANCE,
             HIT_SBT_STRIDE_IGNORE_GEOMETRY, MISS_TYPE_RADIANCE, ray, payload);

    hitValue += payload.hitValue * curWeight;
    curWeight *= payload.weight;
  }

  if (registers.frame == 0) {
    image[imageCoords] = float4(hitValue, 1);
    return;
  }

  float a = 1.0 / float(registers.frame + 1);
  float3 oldColour = image[imageCoords].rgb;
  float3 newColour = lerp(oldColour, hitValue, a);

  image[imageCoords] = float4(newColour, 1);
}
