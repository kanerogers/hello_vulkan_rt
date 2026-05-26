import common;

[shader("miss")]
void main(inout HitPayload payload)
{
    if (payload.depth == 0)
    {
        payload.hitValue = float3(0.1, 0.1, 0.2) * 0.8;
    }
    else
    {
        payload.hitValue = float3(0.001);
    }

    payload.depth = 100;
}

