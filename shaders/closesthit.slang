import common;
import sampling;

[shader("closesthit")]
void main(inout HitPayload payload,
          BuiltInTriangleIntersectionAttributes attributes) {
  Primitive primitive = registers.primitiveBuffer[InstanceIndex()];
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

  let panelPhase = frac(uv.y / registers.tunnelBayLengthMetres);
  let panelBand = panelPhase < 0.0100 ? 0.05 : 1.0;
  let albedo = baseAlbedo * panelBand;

  float3 brdf = albedo / M_PI;

  payload.rayOrigin = worldPosition;
  payload.rayDirection = rayDirection;
  // payload.hitValue = emittance;
  // payload.weight = brdf * cosTheta / p;

  // Fake lighting for now
  let lightDir = normalize(float3(0.25, 0.75, 0.35));
  let ndotl = saturate(dot(worldNormal, lightDir));
  payload.hitValue = albedo * (0.08 + 0.75 * ndotl);
  payload.weight = float3(0.0);
  payload.depth = 100;
}
