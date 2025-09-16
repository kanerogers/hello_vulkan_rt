#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require

#include "common.glsl"
#include "sampling.glsl"

hitAttributeEXT vec2 attribs;

layout(location = 0) rayPayloadInEXT HitPayload payload;

void main() {
    Primitive primitive = registers.primitiveBuffer.primitives[gl_InstanceCustomIndexEXT];
    IndexBuffer i = primitive.indexBuffer;
    uint indexOffset = 3 * gl_PrimitiveID;

    ivec3 triangleIndex = ivec3(i.indices[indexOffset], i.indices[indexOffset + 1], i.indices[indexOffset + 2]);

    // Barycentrics
    const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    // Vertices
    const Vertex v0 = primitive.vertexBuffer.vertices[triangleIndex.x];
    const Vertex v1 = primitive.vertexBuffer.vertices[triangleIndex.y];
    const Vertex v2 = primitive.vertexBuffer.vertices[triangleIndex.z];

    // Position
    const vec3 pos0 = v0.position;
    const vec3 pos1 = v1.position;
    const vec3 pos2 = v2.position;
    const vec3 position = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;
    const vec3 world_position = vec3(gl_ObjectToWorldEXT * vec4(position, 1.0));

    // Normal
    const vec3 normal0 = v0.normal;
    const vec3 normal1 = v1.normal;
    const vec3 normal2 = v2.normal;
    const vec3 normal = normal0 * barycentrics.x + normal1 * barycentrics.y + normal2 * barycentrics.z;
    const vec3 world_normal = normalize(vec3(normal * gl_WorldToObjectEXT));

    Material material = primitive.material;
    vec3 emittance = material.emissiveColourFactor * 0.1;

    // Pick a random direction from here and keep going.
    vec3 tangent, bitangent;
    createCoordinateSystem(world_normal, tangent, bitangent);

    vec3 rayOrigin = world_position;
    vec3 rayDirection = samplingHemisphere(payload.seed, tangent, bitangent, world_normal);

    // PDF of samplingHemisphere choosing this rayDirection
    const float cos_theta = dot(rayDirection, world_normal);
    const float p = cos_theta / M_PI;

    // BRDF
    vec3 albedo = material.baseColourFactor.rgb;
    vec3 BRDF = albedo / M_PI;

    payload.rayOrigin = rayOrigin;
    payload.rayDirection = rayDirection;
    payload.hitValue = emittance;
    payload.weight = BRDF * cos_theta / p;
}
