import common;
import sampling;

[shader("closesthit")]
void main(inout HitPayload payload,
          BuiltInTriangleIntersectionAttributes attributes) {
  // Fetch the primitive, based on bayId and geometryKind
  let bayId = InstanceID();
  let geometryKind = GeometryIndex();
  let primitiveIndex = bayId * BAY_GEOMETRY_KIND_COUNT + geometryKind;

  Primitive primitive = registers.primitiveBuffer[primitiveIndex];

  uint indexOffset = 3 * PrimitiveIndex();

  uint3 triangleIndex = uint3(primitive.indexBuffer[indexOffset],
                              primitive.indexBuffer[indexOffset + 1],
                              primitive.indexBuffer[indexOffset + 2]);

  float2 attribs = attributes.barycentrics;
  float3 barycentrics =
      float3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

  // Extract the vertices
  Vertex v0 = primitive.vertexBuffer[triangleIndex.x];
  Vertex v1 = primitive.vertexBuffer[triangleIndex.y];
  Vertex v2 = primitive.vertexBuffer[triangleIndex.z];

  // Get the uv
  let uv =
      v0.uv * barycentrics.x + v1.uv * barycentrics.y + v2.uv * barycentrics.z;

  // Position
  let position = v0.position * barycentrics.x + v1.position * barycentrics.y +
                 v2.position * barycentrics.z;

  float3x4 objectToWorld = ObjectToWorld3x4();
  float3 worldPosition = mul(objectToWorld, float4(position, 1.0));

  float3 normal = v0.normal * barycentrics.x + v1.normal * barycentrics.y +
                  v2.normal * barycentrics.z;

  float3x4 worldToObject = WorldToObject3x4();
  float4 transformedNormal = mul(normal, worldToObject);
  float3 worldNormal = normalize(transformedNormal.xyz);

  Material material = *primitive.material;
  float3 emittance = material.emissiveColourFactor * 0.1;

  float3 tangent;
  float3 bitangent;
  createCoordinateSystem(worldNormal, tangent, bitangent);

  float3 rayDirection =
      samplingHemisphere(payload.seed, tangent, bitangent, worldNormal);
  float cosTheta = dot(rayDirection, worldNormal);
  float p = cosTheta / M_PI;

  let baseAlbedo = saturate(material.baseColourFactor.rgb);

  var panelBand = 1.;

  if (geometryKind == GEOMETRY_KIND_TUNNEL_SHELL) {
    let panelPhase = frac(uv.y / registers.tunnelBayLengthMetres);
    panelBand = panelPhase < 0.0100 ? 0.05 : 1.0;
  }

  let albedo = saturate(baseAlbedo * panelBand);

  float3 brdf = albedo / M_PI;

  payload.rayOrigin = worldPosition;
  payload.rayDirection = rayDirection;
  // payload.hitValue = emittance;
  // payload.weight = brdf * cosTheta / p;

  let emissive = material.emissiveColourFactor;

  var directLight = float3(0.0);

  let activeRangeMetres = 80.0;

  let bayCount = max(registers.lightCount / TUNNEL_LIGHTS_PER_BAY, 1u);

  let previousBay = bayId == 0 ? bayCount - 1 : bayId - 1;
  let nextBay = bayId + 1 == bayCount ? 0 : bayId + 1;

  directLight += sampleTunnelLight(previousBay * TUNNEL_LIGHTS_PER_BAY +
                                       LED_LIGHT_OFFSET_IN_BAY,
                                   worldPosition, worldNormal);

  directLight +=
      sampleTunnelLight(bayId * TUNNEL_LIGHTS_PER_BAY + LED_LIGHT_OFFSET_IN_BAY,
                        worldPosition, worldNormal);

  directLight += sampleTunnelLight(nextBay * TUNNEL_LIGHTS_PER_BAY +
                                       LED_LIGHT_OFFSET_IN_BAY,
                                   worldPosition, worldNormal);

  let stripBase =
      bayId * TUNNEL_LIGHTS_PER_BAY + LOWER_STRIP_LIGHT_OFFSET_IN_BAY;

  for (uint stripLightIndex = 0; stripLightIndex < LOWER_STRIP_LIGHTS_PER_BAY;
       stripLightIndex++) {
    directLight += sampleTunnelLight(stripBase + stripLightIndex, worldPosition,
                                     worldNormal);
  }

  payload.hitValue = emissive + albedo * (0.01 + directLight);
  payload.weight = float3(0.0);
  payload.depth = 100;

  // let bayDebug = frac(float(bayId) * 0.137);
  // payload.hitValue = float3(bayDebug, 1.0 - bayDebug, 0.25) * directLight;
  // payload.weight = float3(0.0);
  // payload.depth = 100;
  return;
}

float3 sampleTunnelLight(uint lightIndex, float3 worldPosition,
                         float3 worldNormal) {
  if (lightIndex >= registers.lightCount) {
    return float3(0.0);
  }

  let light = registers.lightBuffer[lightIndex];

  let toLight = light.position - worldPosition;
  let distanceMetres = length(toLight);
  let lightDir = toLight / max(distanceMetres, 0.001);

  let ndotl = saturate(dot(worldNormal, lightDir));
  let radiusFade = saturate(1.0 - distanceMetres / light.radiusMetres);
  let attenuation = radiusFade * radiusFade;

  if (ndotl <= 0.0 || attenuation <= 0.0) {
    return float3(0.0);
  }

  RayDesc shadowRay;
  shadowRay.Origin = worldPosition + worldNormal * 0.01;
  shadowRay.Direction = lightDir;
  shadowRay.TMin = 0.001;
  shadowRay.TMax = max(distanceMetres - 1.0, 0.001);

  ShadowPayload shadowPayload;
  shadowPayload.hit = true;

  TraceRay(tlas,
           RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE |
               RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
           0xFF, RAY_TYPE_RADIANCE, HIT_SBT_STRIDE_IGNORE_GEOMETRY,
           MISS_TYPE_SHADOW, shadowRay, shadowPayload);

  if (shadowPayload.hit) {
    return float3(0.0);
  }

  return light.colour * light.intensity * ndotl * attenuation;
}
