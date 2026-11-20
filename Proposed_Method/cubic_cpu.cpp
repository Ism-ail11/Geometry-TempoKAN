#include <algorithm>
#include <cmath>
#include <cstdint>

static float position(float x, int grid) {
    return (std::max(-3.0f, std::min(3.0f, x)) + 3.0f) * (grid / 6.0f);
}

extern "C" void tempo_cubic_forward(
    const float* signal, float* basis, int64_t size, int grid, int with_base
) {
    const int width = grid + 3;
    const int stride = width + with_base;
    for (int64_t i = 0; i < size; i++) {
        const float x = signal[i];
        const float p = position(x, grid);
        const int offset = std::min(static_cast<int>(p), grid - 1);
        const float u = p - offset;
        const float v = 1.0f - u;
        const float u2 = u * u;
        float* out = basis + i * stride;
        if (with_base) *out++ = x;
        for (int k = 0; k < width; k++) out[k] = 0.0f;
        out[offset] = v * v * v / 6.0f;
        out[offset + 1] = (4.0f + (3.0f * u - 6.0f) * u2) / 6.0f;
        out[offset + 2] = (1.0f + 3.0f * u + (3.0f - 3.0f * u) * u2) / 6.0f;
        out[offset + 3] = u2 * u / 6.0f;
    }
}

extern "C" void tempo_cubic_backward(
    const float* signal, const float* gradients, float* out, int64_t size, int grid, int with_base
) {
    const int stride = grid + 3 + with_base;
    for (int64_t i = 0; i < size; i++) {
        const float x = signal[i];
        const float* grad = gradients + i * stride;
        float result = with_base ? *grad++ : 0.0f;
        if (x >= -3.0f && x <= 3.0f) {
            const float p = position(x, grid);
            const int offset = std::min(static_cast<int>(p), grid - 1);
            const float u = p - offset;
            const float v = 1.0f - u;
            const float u2 = u * u;
            grad += offset;
            result += (grid / 6.0f) * (
                -0.5f * v * v * grad[0] + (1.5f * u2 - 2.0f * u) * grad[1]
                + (0.5f + u - 1.5f * u2) * grad[2] + 0.5f * u2 * grad[3]
            );
        }
        out[i] = result;
    }
}
