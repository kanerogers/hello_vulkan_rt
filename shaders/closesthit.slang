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
  Material material = *primitive.material;

  let emissive = material.emissiveColourFactor;

  // If this is emissive, ie. a light, nothing more to do
  if (isEmissiveGeometry(geometryKind)) {
    payload.hitValue = emissive;
    payload.weight = float3(0.0);
    payload.depth = 100;
    return;
  }

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

  // Get the UV
  var uv =
      v0.uv * barycentrics.x + v1.uv * barycentrics.y + v2.uv * barycentrics.z;

  if (geometryKind == GEOMETRY_KIND_TUNNEL_SHELL) {
    uv *= 0.1;
  }

  if (geometryKind == GEOMETRY_KIND_SERVICE_WALKWAY) {
    uv *= 0.4;
  }

  if (geometryKind == GEOMETRY_KIND_SLAB_BED) {
    uv *= 0.1;
  }

  if (geometryKind == GEOMETRY_KIND_RAILS) {
    uv *= 0.1;
  }

  // Position
  let position = v0.position * barycentrics.x + v1.position * barycentrics.y +
                 v2.position * barycentrics.z;

  float3x4 objectToWorld = ObjectToWorld3x4();
  float3 worldPosition = mul(objectToWorld, float4(position, 1.0));

  let worldP0 = mul(objectToWorld, float4(v0.position, 1.0)).xyz;
  let worldP1 = mul(objectToWorld, float4(v1.position, 1.0)).xyz;
  let worldP2 = mul(objectToWorld, float4(v2.position, 1.0)).xyz;

  float3 normal = v0.normal * barycentrics.x + v1.normal * barycentrics.y +
                  v2.normal * barycentrics.z;

  float3x4 worldToObject = WorldToObject3x4();
  float4 transformedNormal = mul(normal, worldToObject);
  float3 worldNormal = normalize(transformedNormal.xyz);

  worldNormal = applyNormalMap(material, uv, worldNormal, worldP0, worldP1,
                               worldP2, v0.uv, v1.uv, v2.uv);

  var baseAlbedo = material.baseColourFactor.rgb;

  if (material.baseColourTextureID != NO_TEXTURE) {
    let sampledBaseColour =
        textures[NonUniformResourceIndex(material.baseColourTextureID)]
            .SampleLevelZero(uv, 0)
            .rgb;

    baseAlbedo *= sampledBaseColour;
  }

  var albedo = saturate(baseAlbedo);

  var ambientOcclusion = 1.0;
  var roughness = 0.85;
  var metalness = 0.0;

  if (material.metallicRoughnessTextureID != NO_TEXTURE) {
    let orm =
        textures[NonUniformResourceIndex(material.metallicRoughnessTextureID)]
            .SampleLevelZero(uv, 0)
            .rgb;

    ambientOcclusion = orm.r;
    roughness = saturate(orm.g);
    metalness = saturate(orm.b);
  }

  // let aoDistanceMetres = 2.0f;
  // let aoRayOrigin = worldPosition + worldNormal * 0.01f;
  // let aoRayDirection = normalize(worldNormal + float3(0.37f, 0.61f, 0.19f));

  // let aoVisible =
  //     traceVisibilityRay(aoRayOrigin, aoRayDirection, 0.02f,
  //     aoDistanceMetres);
  // let ao = aoVisible ? 1.0f : 0.55f;

  let diffuseAlbedo = albedo * (1.0 - metalness);
  let ambientTerm = 0.001 * ambientOcclusion;

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

  if (receivesLowerStripLighting(geometryKind)) {
    let stripBase =
        bayId * TUNNEL_LIGHTS_PER_BAY + LOWER_STRIP_LIGHT_OFFSET_IN_BAY;

    for (uint stripLightIndex = 0; stripLightIndex < LOWER_STRIP_LIGHTS_PER_BAY;
         stripLightIndex++) {
      directLight += sampleTunnelLight(stripBase + stripLightIndex,
                                       worldPosition, worldNormal);
    }
  }

  float3 finalColour = emissive + albedo * (ambientTerm + directLight);

  payload.hitValue = finalColour;
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

  let distance01 = saturate(distanceMetres / light.radiusMetres);
  let radiusFade = saturate(1.0 - distance01 * distance01);
  let distanceAttenuation = radiusFade * radiusFade;

  let fromLightToSurface = -lightDir;
  let emissionCos =
      saturate(dot(normalize(light.emissionDirection), fromLightToSurface));
  let directionalAttenuation = pow(emissionCos, light.directionalFalloffPower);

  let attenuation = distanceAttenuation * directionalAttenuation;

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

bool isEmissiveGeometry(uint geometryKind) {
  return geometryKind == GEOMETRY_KIND_LED_TUBES ||
         geometryKind == GEOMETRY_KIND_LOWER_STRIPS;
}

bool receivesLowerStripLighting(uint geometryKind) {
  return geometryKind == GEOMETRY_KIND_TUNNEL_SHELL ||
         geometryKind == GEOMETRY_KIND_SLAB_BED ||
         geometryKind == GEOMETRY_KIND_RAILS ||
         geometryKind == GEOMETRY_KIND_SERVICE_WALKWAY;
}

float3 applyNormalMap(Material material, float2 uv, float3 worldNormal,
                      float3 worldP0, float3 worldP1, float3 worldP2,
                      float2 uv0, float2 uv1, float2 uv2) {
  if (material.normalTextureID == NO_TEXTURE) {
    return worldNormal;
  }

  let dp1 = worldP1 - worldP0;
  let dp2 = worldP2 - worldP0;
  let duv1 = uv1 - uv0;
  let duv2 = uv2 - uv0;

  let det = duv1.x * duv2.y - duv1.y * duv2.x;
  if (abs(det) < 0.000001) {
    return worldNormal;
  }

  let invDet = 1.0 / det;
  var tangent = (dp1 * duv2.y - dp2 * duv1.y) * invDet;
  var bitangent = (dp2 * duv1.x - dp1 * duv2.x) * invDet;

  tangent = normalize(tangent - worldNormal * dot(worldNormal, tangent));

  let handedness =
      dot(cross(tangent, bitangent), worldNormal) < 0.0 ? -1.0 : 1.0;
  bitangent = normalize(cross(worldNormal, tangent) * handedness);

  var tangentNormal =
      textures[NonUniformResourceIndex(material.normalTextureID)]
              .SampleLevelZero(uv, 0)
              .xyz *
          2.0 -
      1.0;

  return normalize(tangent * tangentNormal.x + bitangent * tangentNormal.y +
                   worldNormal * tangentNormal.z);
}

bool traceVisibilityRay(float3 origin, float3 direction, float rayTMin,
                        float rayTMax) {
  if (rayTMax <= rayTMin) {
    return true;
  }

  RayDesc ray;
  ray.Origin = origin;
  ray.Direction = normalize(direction);
  ray.TMin = rayTMin;
  ray.TMax = rayTMax;

  ShadowPayload payload;
  payload.hit = true;

  TraceRay(tlas,
           RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE |
               RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
           0xFF, RAY_TYPE_RADIANCE, HIT_SBT_STRIDE_IGNORE_GEOMETRY,
           MISS_TYPE_SHADOW, ray, payload);

  return !payload.hit;
}
