#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <cmath>

void saveReconstructionGrid(const std::vector<std::vector<double>>& originals, const std::vector<std::vector<double>>& reconstructions, const std::string& path) {
    if (originals.size() != reconstructions.size()) {
        throw std::invalid_argument("length mismatch between originals and reconstructions");
    }

    const int n = static_cast<int>(originals.size());
    const int out_w = n * 28 + (n - 1) * 4;
    const int out_h = 60;
    const int pixels = 784;

    std::vector<uint8_t> canvas(out_w * out_h, 180);

    // calculates pixel position and then writes it to canvas
    auto paint = [&](const std::vector<double>& img, int col, int offset) {
        int start = col * 32;
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                double v = img[i * 28 + j];
                uint8_t byte = static_cast<uint8_t>(std::min(255.0, std::max(0.0, v * 255.0))); // determines 'intensity' based off probability value
                int x = start + j;
                int y = offset + i;
                canvas[y * out_w + x] = byte;
            }
        }
    };

    for (int i = 0; i < n; i++) {
        paint(originals[i], i, 0); // originals
        paint(reconstructions[i], i, 28 + 4); // reconstructions
    }

    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("could not open file for writing: " + path);
    }

    f << "P5\n" << out_w << " " << out_h << "\n" << "255\n";
    f.write(reinterpret_cast<const char*>(canvas.data()), canvas.size());
}