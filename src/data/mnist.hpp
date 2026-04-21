#pragma once

#include <vector>
#include <string>

class MNISTloader {
public:
    MNISTloader(const std::string& img_pth, const std::string& lbl_pth);

    std::vector<std::vector<double>> images;
    std::vector<int> labels;

    int numImages() const { 
        return images.size(); 
    }

private:
    int reverse(int i);
    double toBin(int pixel);
};