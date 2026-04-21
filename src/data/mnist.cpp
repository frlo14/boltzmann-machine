#include "mnist.hpp"
#include <fstream>
#include <stdexcept>

// need to flip so images are compatible with x86 systems
int MNISTloader::reverse(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

double MNISTloader::toBin(int pixel) {
    return pixel > 128 ? 1.0 : 0.0;
}

MNISTloader::MNISTloader(const std::string& img_pth, const std::string& lbl_pth) {
    std::ifstream imageFile(img_pth, std::ios::binary);
    if (!imageFile.is_open())
        throw std::runtime_error("could not open image file: " + img_pth);

    int magic_num, ttl_images, rows, cols;
    imageFile.read((char*)&magic_num, sizeof(magic_num));
    imageFile.read((char*)&ttl_images, sizeof(ttl_images));
    imageFile.read((char*)&rows, sizeof(rows));
    imageFile.read((char*)&cols, sizeof(cols));

    magic_num = reverse(magic_num);
    ttl_images = reverse(ttl_images);
    rows = reverse(rows);
    cols = reverse(cols);

    int pixels = rows * cols;  

    images.resize(ttl_images, std::vector<double>(pixels));
    for (int i = 0; i < ttl_images; i++) {
        for (int p = 0; p < pixels; p++) {
            unsigned char pixel;
            imageFile.read((char*)&pixel, sizeof(pixel));
            images[i][p] = toBin((int)pixel);
        }
    }

    std::ifstream labelFile(lbl_pth, std::ios::binary);
    if (!labelFile.is_open())
        throw std::runtime_error("could not open label file: " + lbl_pth);

    int magic_num_lbl, ttl_labels;
    labelFile.read((char*)&magic_num_lbl, sizeof(magic_num_lbl));
    labelFile.read((char*)&ttl_labels, sizeof(ttl_labels));

    magic_num_lbl = reverse(magic_num_lbl);
    ttl_labels = reverse(ttl_labels);

    labels.resize(ttl_labels);
    for (int i = 0; i < ttl_labels; i++) {
        unsigned char label;
        labelFile.read((char*)&label, sizeof(label));
        labels[i] = (int)label;
    }
}