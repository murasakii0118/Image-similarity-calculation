#include<opencv2/opencv.hpp>
#include<iostream>
#include<bitset>
#include<algorithm>
#include<execution>
#include<vector>


uint32_t distance(std::bitset<64> img1,std::bitset<64> img2){
    auto xor_result = img1 ^ img2;
    return xor_result.count();
}

double cosine_similarity(const std::bitset<64>& hash1, const std::bitset<64>& hash2) {
    double dot = 0.0;
    double norm1 = 0.0;
    double norm2 = 0.0;
    
    for (int i = 0; i < 64; i++) {
        double v1 = hash1[i] ? 1.0 : 0.0;
        double v2 = hash2[i] ? 1.0 : 0.0;
        dot += v1 * v2;
        norm1 += v1 * v1;
        norm2 += v2 * v2;
    }
    return dot / (sqrt(norm1) * sqrt(norm2));
}

std::bitset<64> aHash(cv::Mat in) {
    std::bitset<64> Hash;
    cv::Mat img;
    cv::resize(in, img, {8, 8});
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::Scalar avg = cv::mean(img);
    
    // 串行版本
    for (int i = 0; i < 64; i++) {
        Hash[i] = (img.at<uchar>(i / 8, i % 8) > avg[0]);
    }
    
    return Hash;
}

std::bitset<64> dHash(cv::Mat in) {
    std::bitset<64> Hash;
    cv::Mat img;
    cv::resize(in, img, {9, 8});
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

    // 串行版本
    for (int i = 0; i < 64; i++) {
        int row = i / 8;
        int col = i % 8;
        Hash[i] = (img.at<uchar>(row, col) > img.at<uchar>(row, col + 1));
    }
    
    return Hash;
}

std::bitset<64> pHash(cv::Mat in) {
    std::bitset<64> Hash;
    cv::Mat img;
    cv::resize(in, img, {32, 32});
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    img.convertTo(img, CV_32F);
    cv::Mat dct_img;
    cv::dct(img, dct_img);
    cv::Mat low_freq = dct_img(cv::Rect(0, 0, 8, 8));
    
    // 计算平均值（跳过直流分量）
    double sum = 0;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (i == 0 && j == 0) continue;
            sum += low_freq.at<float>(i, j);
        }
    }
    float avg_value = sum / 63.0f;
    
    // 串行版本
    for (int i = 0; i < 64; i++) {
        int row = i / 8;
        int col = i % 8;
        if (row == 0 && col == 0) {
            Hash[i] = 0;
            continue;
        }
        Hash[i] = (low_freq.at<float>(row, col) > avg_value);
    }
    
    return Hash;
}

std::bitset<64> wHash(cv::Mat in) {
    std::bitset<64> Hash;
    cv::Mat img;
    cv::resize(in, img, {32, 32});
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    img.convertTo(img, CV_32F);

    int rows = img.rows;
    int cols = img.cols;
    
    // 水平方向小波变换
    for (int i = 0; i < rows; i++) {
        cv::Mat row = img.row(i);
        cv::Mat temp = row.clone();
        for (int j = 0; j < cols / 2; j++) {
            float a = temp.at<float>(0, 2 * j);
            float b = temp.at<float>(0, 2 * j + 1);
            row.at<float>(0, j) = (a + b) / 2.0f;          // 低频（近似系数）
            row.at<float>(0, j + cols / 2) = (a - b) / 2.0f; // 高频（细节系数）
        }
    }
    
    // 垂直方向小波变换
    for (int j = 0; j < cols; j++) {
        cv::Mat col = img.col(j);
        cv::Mat temp = col.clone();
        for (int i = 0; i < rows / 2; i++) {
            float a = temp.at<float>(2 * i, 0);
            float b = temp.at<float>(2 * i + 1, 0);
            col.at<float>(i, 0) = (a + b) / 2.0f;          // 低频（近似系数）
            col.at<float>(i + rows / 2, 0) = (a - b) / 2.0f; // 高频（细节系数）
        }
    }
    
    // 5. 取左上角8x8的低频部分（LL子带）
    cv::Mat low_freq = img(cv::Rect(0, 0, 8, 8));
    
    // 6. 计算平均值
    cv::Scalar avg = cv::mean(low_freq);
    float avg_value = avg[0];
    
    // 7. 生成哈希值
    for (int i = 0; i < 64; i++) {
        int row = i / 8;
        int col = i % 8;
        Hash[i] = (low_freq.at<float>(row, col) > avg_value);
    }
    
    return Hash;
}

std::bitset<64> mHash(cv::Mat in) {
    std::bitset<64> Hash;
    cv::Mat img;
    
    cv::resize(in, img, {32, 32});
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    img.convertTo(img, CV_32F);
    
    // 高斯模糊
    cv::Mat blurred;
    cv::GaussianBlur(img, blurred, cv::Size(5, 5), 1.0);
    
    // LoG滤波
    cv::Mat log_kernel = (cv::Mat_<float>(5, 5) <<
        0,  0, -1,  0,  0,
        0, -1, -2, -1,  0,
       -1, -2, 16, -2, -1,
        0, -1, -2, -1,  0,
        0,  0, -1,  0,  0);
    
    cv::Mat log_img;
    cv::filter2D(blurred, log_img, CV_32F, log_kernel);
    
    // 缩放为8x8
    cv::Mat resized;
    cv::resize(log_img, resized, {8, 8});
    
    // 计算哈希
    cv::Scalar avg = cv::mean(resized);
    float avg_value = avg[0];
    
    for (int i = 0; i < 64; i++) {
        int row = i / 8;
        int col = i % 8;
        Hash[i] = (resized.at<float>(row, col) > avg_value);
    }
    
    return Hash;
}

std::bitset<64> rHash(cv::Mat in) {
    std::bitset<64> Hash;
    cv::Mat img;
    cv::resize(in, img, {32, 32});
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    img.convertTo(img, CV_32F);
    cv::normalize(img, img, 0, 1, cv::NORM_MINMAX);
    // 使用多个角度的投影
    int num_angles = 8;  // 使用8个角度
    cv::Mat projections = cv::Mat::zeros(num_angles, 32, CV_32F);
    for (int i = 0; i < num_angles; i++) {
        double angle = i * 180.0 / num_angles;
        // 旋转图像
        cv::Point2f center(16, 16);
        cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);
        cv::Mat rotated;
        cv::warpAffine(img, rotated, rot_mat, img.size());
        
        // 计算每行的平均值（投影）
        for (int row = 0; row < 32; row++) {
            cv::Mat row_data = rotated.row(row);
            projections.at<float>(i, row) = cv::mean(row_data)[0];
        }
    }
    
    // 对投影结果进行DCT
    cv::Mat proj_dct;
    cv::dct(projections, proj_dct);
    cv::Mat low_freq = cv::Mat::zeros(8, 8, CV_32F);
    int rows = std::min(8, proj_dct.rows);
    int cols = std::min(8, proj_dct.cols);
    proj_dct(cv::Rect(0, 0, cols, rows)).copyTo(low_freq(cv::Rect(0, 0, cols, rows)));
    cv::Scalar avg = cv::mean(low_freq);    
    for (int i = 0; i < 64; i++) {
        int row = i / 8;
        int col = i % 8;
        Hash[i] = (low_freq.at<float>(row, col) > avg[0]);
    }
    
    return Hash;
}

std::bitset<64> cmHash(cv::Mat in) {
    std::bitset<64> Hash;
    cv::Mat img;
    cv::resize(in, img, {8, 8});
    cv::cvtColor(img, img, cv::COLOR_BGR2HSV);
    
    // 3. 分离HSV通道
    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    
    // 4. 计算颜色矩特征
    std::vector<float> features;
    
    for (int c = 0; c < 3; c++) {
        cv::Mat channel = channels[c];
        channel.convertTo(channel, CV_32F);
        
        // 一阶矩（均值）
        cv::Scalar mean = cv::mean(channel);
        float m1 = static_cast<float>(mean[0]);
        features.push_back(m1);
        
        // 二阶矩（标准差）
        cv::Mat diff = channel - m1;
        cv::Mat diff_sq;
        cv::multiply(diff, diff, diff_sq);
        cv::Scalar var = cv::mean(diff_sq);
        float m2 = std::sqrt(static_cast<float>(var[0]));
        features.push_back(m2);
        
        // 三阶矩（偏度）
        if (m2 > 0.001f) {
            cv::Mat diff_cube;
            cv::pow(diff, 3, diff_cube);
            cv::Scalar third = cv::mean(diff_cube);
            float m3 = static_cast<float>(third[0]) / (m2 * m2 * m2);
            features.push_back(m3);
        } else {
            features.push_back(0.0f);
        }
    }
    
    //直接扩展特征到64维
    cv::Mat extended(1, 64, CV_32F);
    
    // 使用插值扩展（更平滑）
    // cv::Mat small_mat(1, 9, CV_32F, features.data());
    // cv::resize(small_mat, extended, cv::Size(64, 1), 0, 0, cv::INTER_LINEAR);
    
    for (int i = 0; i < 64; i++) {
        extended.at<float>(0, i) = features[i % features.size()];
    }
    // 6. 归一化
    cv::normalize(extended, extended, 0, 1, cv::NORM_MINMAX);
    
    // 7. 生成哈希
    cv::Scalar avg = cv::mean(extended);    
    for (int i = 0; i < 64; i++) {
        Hash[i] = (extended.at<float>(0, i) > avg[0]);
    }
    
    return Hash;
}

bool similarity_probability(cv::Mat &img1,cv::Mat &img2){
    auto a1 = aHash(img1);
    auto d1 = dHash(img1);
    auto p1 = pHash(img1);
    auto w1 = wHash(img1);
    auto m1 = mHash(img1);
    auto r1 = rHash(img1);
    auto cm1 = cmHash(img1);

    auto a2 = aHash(img2);
    auto d2 = dHash(img2);
    auto p2 = pHash(img2);
    auto w2 = wHash(img2);
    auto m2 = mHash(img2);
    auto r2 = rHash(img2);
    auto cm2 = cmHash(img2);
    uint8_t vote = 0;
    if(cosine_similarity(a1,a2) > 0.7) vote++;
    if(cosine_similarity(d1,d2) > 0.7) vote++;
    if(cosine_similarity(p1,p2) > 0.7) vote++;
    if(cosine_similarity(w1,w2) > 0.7) vote++;
    if(cosine_similarity(m1,m2) > 0.7) vote++;
    if(cosine_similarity(r1,r2) > 0.7) vote++;
    if(cosine_similarity(cm1,cm2) > 0.7) vote++;
    if(vote>=4) return true;
    return false;

}

// int main(int argc, char const *argv[]){
//     std::string in;
//     std::cin >> in;
//     std::string inn;
//     std::cin >> inn;
//     auto img1 = cv::imread(in);
//     auto a1 = aHash(img1);
//     auto d1 = dHash(img1);
//     auto p1 = pHash(img1);
//     auto w1 = wHash(img1);
//     auto m1 = mHash(img1);
//     auto r1 = rHash(img1);
//     auto cm1 = cmHash(img1);
//     std::cout << a1.to_string() << "\n" 
//         << d1.to_string() << "\n" 
//         << p1.to_string() << "\n" 
//         << w1.to_string() << "\n" 
//         << m1.to_string() << "\n" 
//         << r1.to_string() << "\n" 
//         << cm1.to_string() <<"\n";
//     auto img = cv::imread(inn);
//     auto a2 = aHash(img);
//     auto d2 = dHash(img);
//     auto p2 = pHash(img);
//     auto w2 = wHash(img);
//     auto m2 = mHash(img);
//     auto r2 = rHash(img);
//     auto cm2 = cmHash(img);
//     std::cout << a2.to_string() << "\n" 
//         << d2.to_string() << "\n" 
//         << p2.to_string() << "\n" 
//         << w2.to_string() << "\n" 
//         << m2.to_string() << "\n" 
//         << r2.to_string() << "\n" 
//         << cm2.to_string() << "\n\n";
//     std::cout << "dist:" << distance(a1,a2) << "/" 
//         << distance(d1,d2) << "/"
//         << distance(p1,p2) << "/" 
//         << distance(w1,w2) <<"/" 
//         << distance(m1,m2) << "/" 
//         << distance(r1,r2) << "/" 
//         << distance(cm1,cm2) <<  "/\n";
//     std::cout << "CoSimilar:" << cosine_similarity(a1,a2) << "/" 
//         << cosine_similarity(d1,d2) << "/"
//         << cosine_similarity(p1,p2) << "/" 
//         << cosine_similarity(w1,w2) <<"/" 
//         << cosine_similarity(m1,m2) << "/" 
//         << cosine_similarity(r1,r2) << "/" 
//         << cosine_similarity(cm1,cm2) <<  "/";
//     return 0;
// }

int main(int argc, char const *argv[])
{
    std::string in;
    std::cin >> in;
    std::string inn;
    std::cin >> inn;
    auto img1 = cv::imread(in);
    auto img2 = cv::imread(inn);
    std::cout << "该图片:" << similarity_probability(img1,img2);
    return 0;
}

