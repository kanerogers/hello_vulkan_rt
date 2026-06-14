static const float M_PI = 3.1415926535897932384626433832795;

uint tea(uint val0, uint val1)
{
    uint v0 = val0;
    uint v1 = val1;
    uint s0 = 0;

    for(uint n = 0; n < 16; n++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    return v0;
}

uint lcg(inout uint prev)
{
    uint LCG_A = 1664525u;
    uint LCG_C = 1013904223u;
    prev = (LCG_A * prev + LCG_C);
    return prev & 0x00FFFFFF;
}

float rnd(inout uint prev)
{
    return float(lcg(prev)) / float(0x01000000);
}

float3 samplingHemisphere(inout uint seed, float3 tangent, float3 bitangent, float3 normal)
{
    float r1 = rnd(seed);
    float r2 = rnd(seed);
    float sq = sqrt(1.0 - r2);

    float3 direction =
        float3(cos(2 * M_PI * r1) * sq, sin(2 * M_PI * r1) * sq, sqrt(r2));
    direction = direction.x * tangent + direction.y * bitangent + direction.z * normal;

    return direction;
}

void createCoordinateSystem(float3 normal, out float3 tangent, out float3 bitangent)
{
    if (abs(normal.x) > abs(normal.y))
    {
        tangent = float3(normal.z, 0, -normal.x) / sqrt(normal.x * normal.x + normal.z * normal.z);
    }
    else
    {
        tangent = float3(0, -normal.z, normal.y) / sqrt(normal.y * normal.y + normal.z * normal.z);
    }

    bitangent = cross(normal, tangent);
}

