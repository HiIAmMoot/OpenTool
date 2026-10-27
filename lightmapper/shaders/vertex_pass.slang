/*
VERTEX_PASS.SLANG

The vertex pass's two ray-query kernels: gather (the luminance of what each cosine-weighted ray of a query sees)
and shadow (the sun and edge rays, with the tint they carry through the panes they pass). Both walk the
pass-through loop of blender/bake.py's VertexPass.cast over one acceleration structure of the receivers, then the
occluders, so a triangle id means the same here as there. GROUP_SIZE, RAY_LENGTH, RAY_EPSILON, PASS_LIMIT and
LUMA arrive as defines from vertex_gpu.py.
*/

/* ---------- scene */

#define FLAG_CUTOUT 1
#define FLAG_CASTS 2
#define FLAG_GLOWS 4
#define FLAG_RECEIVER 8

RaytracingAccelerationStructure world;
/* texture id, flags, page, class */
StructuredBuffer<int4> tri_info;
StructuredBuffer<float3> tri_tint;
StructuredBuffer<float3> tri_emission;
StructuredBuffer<float3> tri_normal;
StructuredBuffer<int> tri_twin;
StructuredBuffer<int3> tri_twin_perm;
StructuredBuffer<uint> tri_vertices;
StructuredBuffer<float2> vertex_tex_uv;
StructuredBuffer<float2> vertex_lm_uv;
/* offset (in texels), width, height */
StructuredBuffer<int4> texture_info;
StructuredBuffer<float4> texture_texels;
StructuredBuffer<int4> page_info;
StructuredBuffer<float3> page_texels;
/* indexed by class + 1 */
StructuredBuffer<float3> ambient;

uniform float albedo_scale;
uniform float albedo_fallback;
uniform int ambient_rows;

/* ---------- gather inputs and outputs */

StructuredBuffer<float3> query_pos;
StructuredBuffer<float3> query_n;
StructuredBuffer<float3> query_tangent;
StructuredBuffer<float3> query_bitangent;
StructuredBuffer<float> query_spin;
StructuredBuffer<int> query_class;
StructuredBuffer<float> hemi_phi;
StructuredBuffer<float> hemi_r;
StructuredBuffer<float> hemi_h;
uniform int rays;
uniform uint gather_count;
RWStructuredBuffer<float> lum_out;

/* ---------- shadow inputs and outputs */

StructuredBuffer<float3> shadow_origin;
StructuredBuffer<float3> shadow_dir;
uniform uint shadow_count;
/* carried rgb and 1 when the ray escaped */
RWStructuredBuffer<float4> shadow_out;

/* ---------- shading */

bool cutout(int tri) { return (tri_info[tri].y & FLAG_CUTOUT) != 0; }
bool casts(int tri) { return (tri_info[tri].y & FLAG_CASTS) != 0; }
bool glows(int tri) { return (tri_info[tri].y & FLAG_GLOWS) != 0; }
bool receiver(int tri) { return (tri_info[tri].y & FLAG_RECEIVER) != 0; }

float3 weights_of(float2 bary)
{
    float3 w = max(float3(1.0 - bary.x - bary.y, bary.x, bary.y), 0.0);
    float total = w.x + w.y + w.z;
    /* (a sliver the hardware reports outside its triangle has no meaningful weights, so its centre stands in) */
    return total > 0 ? w / total : float3(1.0 / 3.0);
}

float2 interpolate(StructuredBuffer<float2> uv, int tri, float3 w)
{
    return uv[tri_vertices[tri * 3]] * w.x + uv[tri_vertices[tri * 3 + 1]] * w.y + uv[tri_vertices[tri * 3 + 2]] * w.z;
}

/* a negative index wraps like numpy's modulo, where Slang's % keeps the sign of the dividend */
int wrapped(int i, int n) { return ((i % n) + n) % n; }

float4 texture_at(int id, float2 uv)
{
    int4 info = texture_info[id];
    float x = uv.x * info.y - 0.5, y = uv.y * info.z - 0.5;
    int x0 = (int)floor(x), y0 = (int)floor(y);
    float fx = x - x0, fy = y - y0;
    int xa = wrapped(x0, info.y), xb = wrapped(x0 + 1, info.y);
    int ya = wrapped(y0, info.z), yb = wrapped(y0 + 1, info.z);
    float4 top = texture_texels[info.x + ya * info.y + xa] * (1 - fx) + texture_texels[info.x + ya * info.y + xb] * fx;
    float4 bottom = texture_texels[info.x + yb * info.y + xa] * (1 - fx)
                    + texture_texels[info.x + yb * info.y + xb] * fx;
    return top * (1 - fy) + bottom * fy;
}

float3 page_at(int id, float2 uv)
{
    int4 info = page_info[id];
    float x = uv.x * info.y - 0.5, y = uv.y * info.z - 0.5;
    int x0 = (int)floor(x), y0 = (int)floor(y);
    float fx = x - x0, fy = y - y0;
    int xa = clamp(x0, 0, info.y - 1), xb = clamp(x0 + 1, 0, info.y - 1);
    int ya = clamp(y0, 0, info.z - 1), yb = clamp(y0 + 1, 0, info.z - 1);
    float3 top = page_texels[info.x + ya * info.y + xa] * (1 - fx) + page_texels[info.x + ya * info.y + xb] * fx;
    float3 bottom = page_texels[info.x + yb * info.y + xa] * (1 - fx) + page_texels[info.x + yb * info.y + xb] * fx;
    return top * (1 - fy) + bottom * fy;
}

float alpha_at(int tri, float2 bary)
{
    return texture_at(tri_info[tri].x, interpolate(vertex_tex_uv, tri, weights_of(bary))).a;
}

/* class -1 (indoor) has its own row; anything below it has none */
float3 ambient_of(int cls)
{
    return cls < -1 ? float3(0.0) : ambient[min(cls + 1, ambient_rows - 1)];
}

/* the luminance of the light arriving along dir at a hit (the CPU pass's shade) */
float shade(int tri, float2 bary, float3 dir, int cls)
{
    float3 w = weights_of(bary);
    int texture = tri_info[tri].x;
    float3 albedo = float3(albedo_fallback);
    if (texture >= 0)
        albedo = texture_at(texture, interpolate(vertex_tex_uv, tri, w)).rgb * albedo_scale;
    if (glows(tri) && !casts(tri))
        albedo = float3(0.0);
    bool facing = dot(tri_normal[tri], dir) < 0;
    if (!receiver(tri))
        return dot(albedo * ambient_of(cls) + tri_emission[tri] * (facing ? 1.0 : 0.0), LUMA);
    if (!facing)
        return 0.0;
    int page = tri_info[tri].z;
    float3 lit = page >= 0 ? page_at(page, interpolate(vertex_lm_uv, tri, w)) : float3(0.0);
    return dot(albedo * (lit + ambient_of(tri_info[tri].w)) + tri_emission[tri], LUMA);
}

/* ---------- the pass-through loop */

int cast(float3 origin, float3 dir, bool shadow, inout float3 carried, out float3 hit, out float2 bary)
{
    float3 start = origin;
    float remaining = RAY_LENGTH;
    hit = origin;
    bary = float2(0.0);
    for (int i = 0; i < PASS_LIMIT + 1; i++) {
        if (remaining <= 0) return -1;
        RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
        RayDesc ray = { start, 0.0, dir, remaining };
        q.TraceRayInline(world, RAY_FLAG_NONE, 0xff, ray);
        while (q.Proceed()) {}
        if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) return -1;
        int tri = q.CommittedPrimitiveIndex();
        float t = q.CommittedRayT();
        bary = q.CommittedTriangleBarycentrics();
        hit = start + t * dir;
        int twin = tri_twin[tri];
        if (twin >= 0 && dot(tri_normal[tri], dir) >= 0 && dot(tri_normal[twin], dir) < 0) {
            float3 w = weights_of(bary);
            int3 perm = tri_twin_perm[tri];
            bary = float2(w[perm.y], w[perm.z]);
            tri = twin;
        }
        bool stops = casts(tri) || (!shadow && glows(tri) && dot(tri_normal[tri], dir) < 0);
        if (stops && !(cutout(tri) && tri_info[tri].x >= 0 && alpha_at(tri, bary) < 0.5)) return tri;
        if (shadow && !casts(tri)) carried *= tri_tint[tri];
        start = hit + dir * RAY_EPSILON;
        remaining -= t + RAY_EPSILON;
    }
    /* (nine passed hits end the ray as a miss, as the cpu pass does) */
    return -1;
}

/* ---------- kernels */

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void gather(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= gather_count) return;
    int query = (int)(i / (uint)rays), k = (int)(i % (uint)rays);
    float angle = hemi_phi[k] + query_spin[query];
    float3 dir = hemi_r[k] * cos(angle) * query_tangent[query] + hemi_r[k] * sin(angle) * query_bitangent[query]
                 + hemi_h[k] * query_n[query];
    float3 carried = float3(1.0), hit;
    float2 bary;
    int tri = cast(query_pos[query] + RAY_EPSILON * dir, dir, false, carried, hit, bary);
    lum_out[i] = tri < 0 ? 0.0 : shade(tri, bary, dir, query_class[query]);
}

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void shadow(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= shadow_count) return;
    float3 carried = float3(1.0), hit;
    float2 bary;
    int tri = cast(shadow_origin[i], shadow_dir[i], true, carried, hit, bary);
    shadow_out[i] = float4(carried, tri < 0 ? 1.0 : 0.0);
}
