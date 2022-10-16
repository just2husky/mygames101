// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector <Eigen::Vector3f> &positions) {
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector <Eigen::Vector3i> &indices) {
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector <Eigen::Vector3f> &cols) {
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f &v3, float w = 1.0f) {
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}


static float cross2d(const Vector2f v1, const Vector2f v2) {
    return v1[0] * v2[1] - v1[1] * v2[0];
}

static bool insideTriangle(double x, double y, const Vector3f* _v)
{
    Eigen::Vector2f p;
    p << x, y;

    Eigen::Vector2f AB, BC, CA;
    AB = _v[0].head(2) - _v[1].head(2);
    BC = _v[1].head(2) - _v[2].head(2);
    CA = _v[2].head(2) - _v[0].head(2);

    Eigen::Vector2f AP, BP, CP;
    AP = _v[0].head(2) - p;
    BP = _v[1].head(2) - p;
    CP = _v[2].head(2) - p;

    return cross2d(AB, AP) > 0 && cross2d(BC, BP) > 0 && cross2d(CA, CP) > 0;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f *v) {
    float c1 = (x * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * y + v[1].x() * v[2].y() - v[2].x() * v[1].y()) /
               (v[0].x() * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * v[0].y() + v[1].x() * v[2].y() -
                v[2].x() * v[1].y());
    float c2 = (x * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * y + v[2].x() * v[0].y() - v[0].x() * v[2].y()) /
               (v[1].x() * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * v[1].y() + v[2].x() * v[0].y() -
                v[0].x() * v[2].y());
    float c3 = (x * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * y + v[0].x() * v[1].y() - v[1].x() * v[0].y()) /
               (v[2].x() * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * v[2].y() + v[0].x() * v[1].y() -
                v[1].x() * v[0].y());
    return {c1, c2, c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type) {
    auto &buf = pos_buf[pos_buffer.pos_id];
    auto &ind = ind_buf[ind_buffer.ind_id];
    auto &col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto &i: ind) {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto &vec: v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto &vert: v) {
            vert.x() = 0.5 * width * (vert.x() + 1.0);
            vert.y() = 0.5 * height * (vert.y() + 1.0);
            vert.z() = -vert.z() * f1 + f2;  // https://games-cn.org/forums/topic/hw2%e7%9a%84%e7%96%91%e9%97%ae/
        }

        for (int i = 0; i < 3; ++i) {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);
        // std::cout << "ori v0, x: " << t.v[0].x() << ", y: " << t.v[0].y() << ", : "<< t.v[0].z() << std::endl;
        // std::cout << "ori v1, x: " << t.v[1].x() << ", y: " << t.v[1].y() << ", : "<< t.v[1].z() << std::endl;
        // std::cout << "ori v2, x: " << t.v[2].x() << ", y: " << t.v[2].y() << ", : "<< t.v[2].z() << std::endl;

        rasterize_triangle(t);
    }
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle &t) {
    auto v = t.toVector4();

    // TODO : Find out the bounding box of current triangle.
    float x_minf = t.v[0].x();
    float x_maxf = t.v[0].x();
    float y_minf = t.v[0].y();
    float y_maxf = t.v[0].y();
    // std::cout << "float x_min: " << x_minf << ", x_max: " << x_maxf << ", y_min: " << y_minf << ", y_max: " << y_maxf << std::endl;


    for (auto &point: t.v) {
        // std::cout << "rasterize_triangle, point" << " " << ", x: " << point.x() << ", y: " << point.y() << std::endl;
        if (point.x() < x_minf) {
            x_minf = point.x();
        }
        if (point.x() > x_maxf) {
            x_maxf = point.x();
        }
        if (point.y() < y_minf) {
            y_minf = point.y();
        }
        if (point.y() > y_maxf) {
            y_maxf = point.y();
        }
    }

    int x_min = ceil(x_minf);
    int x_max = floor(x_maxf);
    int y_min = ceil(y_minf);
    int y_max = floor(y_maxf);
    std::cout << "int x_min: " << x_min << ", x_max: " << x_max << ", y_min: " << y_min << ", y_max: " << y_max << std::endl;

    // iterate through the pixel and find if the current pixel is inside the triangle
    for (int x = x_min; x < x_max; x++) {
        for (int y = y_min; y < y_max; y++) {
            int pixel_x = x + 0.5;
            int pixel_y = y + 0.5;
            bool result = insideTriangle(pixel_x, pixel_y, t.v);
            if (result) {
                // If so, use the following code to get the interpolated z value.
                auto [alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
                float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interpolated =
                        alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                z_interpolated *= w_reciprocal;
                int index = get_index(x, y);
                float depth = depth_buf[index];
                if (z_interpolated < depth) {
                    depth_buf[index] = z_interpolated;
                    // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
                    Eigen::Vector3f point = Eigen::Vector3f(x, y, 0);
                    // Eigen::Vector3f color = Eigen::Vector3f(255, 0, 0);
                    // std::cout << "t.color[0], x: " << t.color[0].x() << ", y: " << t.color[0].y() << ", z: " << t.color[0].z() << std::endl;
                    set_pixel(point, t.getColor());
                }

            }
        }
    }

}

void rst::rasterizer::set_model(const Eigen::Matrix4f &m) {
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f &v) {
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f &p) {
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff) {
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color) {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth) {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h) {
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
}

int rst::rasterizer::get_index(int x, int y) {
    return (height - 1 - y) * width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f &point, const Eigen::Vector3f &color) {
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height - 1 - point.y()) * width + point.x();
    frame_buf[ind] = color;

}

// clang-format on