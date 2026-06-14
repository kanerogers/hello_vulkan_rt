struct VSOut {
    float4 position : SV_Position;
    [[vk::location(0)]]
    float2 uv;
};

[shader("vertex")]
VSOut main(uint vertexId: SV_VertexID)
{
    VSOut output;
    output.uv = float2((vertexId << 1) & 2, vertexId & 2);
    output.position = float4(output.uv * 2.0 - 1.0, 1.0, 1.0);
    return output;
}

