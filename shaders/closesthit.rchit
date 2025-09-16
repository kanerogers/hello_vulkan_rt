#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#include "common.glsl"

hitAttributeEXT vec2 attribs;

layout(location = 0) rayPayloadInEXT vec3 payload;

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

    const vec3 pos0 = v0.position;
    const vec3 pos1 = v1.position;
    const vec3 pos2 = v2.position;

    const vec3 position = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;

    const vec3 normal0 = v0.normal;
    const vec3 normal1 = v1.normal;
    const vec3 normal2 = v2.normal;

    const vec3 normal = normal0 * barycentrics.x + normal1 * barycentrics.y + normal2 * barycentrics.z;
    const vec3 v = vec3(1.0); // who can be bothered doing mathematics
    const float ndotv = dot(normal, v);

    payload = ndotv * primitive.material.baseColourFactor.rgb;
}
