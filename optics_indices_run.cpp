// Copyright (c) 2020 Papa Libasse Sow.
// https://github.com/Nandite/Pcl-Optics
// Distributed under the MIT Software License (X11 license).
//
// SPDX-License-Identifier: MIT
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of
// the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <chrono>
#include <random>
#include "Optics.hpp"


#include <cxxopts.hpp>

//added for filtering
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>



std::tuple<uint8_t, uint8_t, uint8_t> jet(double x){
    const double rone = 0.8;
    const double gone = 1.0;
    const double bone = 1.0;
    double r, g, b;

    x = (x < 0 ? 0 : (x > 1 ? 1 : x));

    if (x < 1. / 8.) {
        r = 0;
        g = 0;
        b = bone * (0.5 + (x) / (1. / 8.) * 0.5);
    } else if (x < 3. / 8.) {
        r = 0;
        g = gone * (x - 1. / 8.) / (3. / 8. - 1. / 8.);
        b = bone;
    } else if (x < 5. / 8.) {
        r = rone * (x - 3. / 8.) / (5. / 8. - 3. / 8.);
        g = gone;
        b = (bone - (x - 3. / 8.) / (5. / 8. - 3. / 8.));
    } else if (x < 7. / 8.) {
        r = rone;
        g = (gone - (x - 5. / 8.) / (7. / 8. - 5. / 8.));
        b = 0;
    } else {
        r = (rone - (x - 7. / 8.) / (1. - 7. / 8.) * 0.5);
        g = 0;
        b = 0;
    }
    return std::make_tuple(uint8_t(255.*r), uint8_t(255.*g), uint8_t(255.*b));
}


//does the prescalling for jet -> maps z to [0-1]:[1-0] in the area between 0 and threshold
//e.g. points along a linear line in z direction would get be: blue, green, yellow, red, yellow, green, blue, green,...
std::tuple<uint8_t, uint8_t, uint8_t> stacked_jet(double z, double threshold){
    pcl::PointXYZRGB pointrgb;
    std::tuple<uint8_t, uint8_t, uint8_t> colors_rgb;
    double r, g, b, val;
    if(z<=0){
        while(z<0){
            z+=threshold;
        }
    }else{
        while(z>threshold){
            z-=threshold;
        }
    }
    if(z>threshold/2){
        z-=(threshold/2);
        val=-((z/(threshold/2))-1);
    }else{
        val=z/(threshold/2);
    }
    //std::cout << "new z: " << z  << "   val: " << val <<std::endl;
    //std::cout << "val: " << val << std::endl;
    //colors_rgb = jet(z/(threshold);
    return jet(val);
    //return std::make_tuple(uint8_t(255.*0), uint8_t(255.*0), uint8_t(255.*0));
}


int main(int argc, char* argv[]) {

    std::string path_str;
    int filter_flag=0;
    int show;
    float filter_leaf_size;
    int max_it, min_pts;
    double distance_threshold, reachability_threshold;
    cxxopts::Options options("MyProgram", "One line description of MyProgram");
    options.add_options()
            ("help", "Print help")
            ("f,filter", "set to 1 for filtering", cxxopts::value<int>(filter_flag)->default_value("0"))
            ("s,show", "show visualization", cxxopts::value<int>(show)->default_value("0"))
            ("input_file", "Input pcd file", cxxopts::value(path_str))
            ("m,max_it", "max iterations", cxxopts::value<int>(max_it)->default_value("100"))
            ("l,filter_leaf_size", "filter_leaf_size", cxxopts::value<float>(filter_leaf_size)->default_value("0.01"))
            ("t,distance_threshold", "distance_threshold", cxxopts::value<double>(distance_threshold)->default_value("0.02"))
            ("p,min_pts", "optics: min_pts", cxxopts::value<int>(min_pts)->default_value("10"))
            ("r,reachability_threshold", "optics: reachability_threshold", cxxopts::value<double>(reachability_threshold)->default_value("0.05"));

    auto result = options.parse(argc, argv);
    if (result.count("help")) {
        cout << options.help({ "", "Group" }) << endl;
        exit(0);
    }

    std::cout << "input_file: " << path_str << std::endl;
    std::cout << "max_it: " << max_it << std::endl;
    std::cout << "distance_threshold: " << distance_threshold << std::endl;
    std::cout << "min_pts: " << min_pts << std::endl;
    std::cout << "reachability_threshold: " << reachability_threshold << std::endl;

    std::mt19937 t(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<unsigned int> rgb(1, 255);

    pcl::PCDReader reader;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>),
            bufferCloud(new pcl::PointCloud<pcl::PointXYZ>),
            filteredCloud(new pcl::PointCloud<pcl::PointXYZ>),
            indicesCloud(new pcl::PointCloud<pcl::PointXYZ>);


    pcl::PointCloud<pcl::PointXYZRGB>::Ptr indicesCloud_RGB(new pcl::PointCloud<pcl::PointXYZRGB>);

    reader.read(path_str, *cloud);
    std::cout << "PointCloud before filtering has: " << cloud->points.size() << " data points." << std::endl;

    //do some filtering on the cloud to remove outliers
    // Create the filtering object for RadiusOutlierRemoval
    pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
    outrem.setRadiusSearch(5.);//good 5 and r = 3//0.8
    outrem.setMinNeighborsInRadius (4);//2
    std::cerr << "Cloud after StatisticalOutlierRemoval: " <<cloud->size()<< std::endl;
    outrem.setInputCloud(cloud);
    outrem.filter (*filteredCloud);
    std::cerr << "Cloud after RadiusOutlierRemoval: " <<filteredCloud->size()<< std::endl;


    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filteredCloud_Color(new pcl::PointCloud<pcl::PointXYZRGB>);
    copyPointCloud(*filteredCloud, *filteredCloud_Color);
    double jet_stacking_threshold=1.;
    for (std::size_t i = 0; i < filteredCloud_Color->points.size (); ++i){
        pcl::PointXYZRGB pointrgb;
        std::tuple<uint8_t, uint8_t, uint8_t> colors_rgb;
        colors_rgb = stacked_jet( filteredCloud_Color->points[i].z, jet_stacking_threshold);

        std::uint32_t rgb = (static_cast<std::uint32_t>(std::get<0>(colors_rgb)) << 16 |
                             static_cast<std::uint32_t>(std::get<1>(colors_rgb)) << 8 |
                             static_cast<std::uint32_t>(std::get<2>(colors_rgb)));
        pointrgb.rgb = *reinterpret_cast<float*>(&rgb);
        filteredCloud_Color->points[i].r = pointrgb.r;
        filteredCloud_Color->points[i].g = pointrgb.g;
        filteredCloud_Color->points[i].b = pointrgb.b;
    }
    std::cerr << "filteredCloud_Color: " <<filteredCloud_Color->size()<< std::endl;

    const unsigned int size = std::floor(filteredCloud_Color->size() / 2);
    pcl::IndicesPtr indices(new std::vector<int>(size));
    std::iota(std::begin(*indices),
              std::end(*indices),
              size);

    std::vector<pcl::PointIndicesPtr> clusters;
    std::cout << "min_pts: " << min_pts << std::endl;
    std::cout << "reachability_threshold: " << reachability_threshold << std::endl;
    //Optics::optics<pcl::PointXYZ>(filteredCloud, indices, min_pts, reachability_threshold, clusters);
    Optics::optics<pcl::PointXYZRGB>(filteredCloud_Color, indices, min_pts, reachability_threshold, clusters);
    std::cout << "Cluster size:" << clusters.size() << std::endl;

    //int v0{0}, v1{1}, v2{2};
    pcl::visualization::PCLVisualizer viewer("Visualizer");
    //sviewer.setSize(1280, 1024);
    viewer.setShowFPS(true);


    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_1(filteredCloud, rgb(t), rgb(t), rgb(t));
    //viewer.addPointCloud(filteredCloud, color_1, "filteredCloud");
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "filteredCloud");


/*
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(filteredCloud_Color);
        extract.setIndices(indices);
        extract.setNegative(false);
        extract.filter(*indicesCloud_RGB);
*/
    pcl::ExtractIndices<pcl::PointXYZRGB> extractRGB;
    extractRGB.setInputCloud(filteredCloud_Color);
    extractRGB.setIndices(indices);
    extractRGB.setNegative(false);
    extractRGB.filter(*indicesCloud_RGB);
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(clusterCloud, rgb(t), rgb(t), rgb(t));
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_2(indicesCloud, rgb(t), rgb(t), rgb(t));
        //viewer.addPointCloud(indicesCloud, color_2, "indicesCloud");
        viewer.addPointCloud(indicesCloud_RGB, "indicesCloud");

      viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "indicesCloud");


    unsigned int id = 0;
    for (const auto &c : clusters) {
        //std::cout << "Cluster " << id << " size is : " << c->indices.size() << std::endl;
        if (c->indices.size() < 10) continue;
        std::cout << "Cluster " << id << " size is : " << c->indices.size() << std::endl;

        pcl::PointCloud<pcl::PointXYZ>::Ptr clusterCloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto &index : c->indices) {
            clusterCloud->push_back((*filteredCloud)[index]);
        }
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(clusterCloud, rgb(t), rgb(t), rgb(t));
        std::string strid = "cloud_cluster_" + std::to_string(id++);
        viewer.addPointCloud(clusterCloud, color, strid);
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, strid);
    }

    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }

    return EXIT_SUCCESS;
}
