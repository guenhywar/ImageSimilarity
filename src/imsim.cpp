#include <uima/api.hpp>

#include <pcl/point_types.h>


//#include <robosherlock/types/all_types.h>
//RS
#include <robosherlock/scene_cas.h>
#include <robosherlock/utils/time.h>
#include <robosherlock/DrawingAnnotator.h>
#include <robosherlock/CASConsumerContext.h>

//rs bs
#include <rs_bs/types/all_types.h>

//imsim
#include <image_similarity/types/similarity_types.h>

//pcl extract indices
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>

//sift pcl
#include <pcl/keypoints/sift_keypoint.h>

#include <set>
#include <map>

//opencv
#include <opencv2/features2d.hpp> //fastfeature detector
#include <opencv2/xfeatures2d/nonfree.hpp> //surf
#include "opencv2/opencv.hpp" //warpperspective


#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include <rs_bs/ViewNames.h>

//icp
#include <pcl/registration/icp.h>

//pcl extract indices
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/fpfh.h> //nicht nötig?
#include <pcl/features/feature.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_rejection_distance.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/keypoints/narf_keypoint.h>
//sift pcl
#include <pcl/keypoints/sift_keypoint.h>


//ab hier die axt
#include <uima/api.hpp>

#include <pcl/point_types.h>


//#include <robosherlock/types/all_types.h>
//RS
#include <robosherlock/scene_cas.h>
#include <robosherlock/utils/time.h>
#include <robosherlock/DrawingAnnotator.h>
#include <robosherlock/CASConsumerContext.h>

//rs bs
#include <rs_bs/types/all_types.h>

//pcl extract indices
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/fpfh.h> //nicht nötig?
#include <pcl/features/feature.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_rejection_distance.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/keypoints/narf_keypoint.h>
//sift pcl
#include <pcl/keypoints/sift_keypoint.h>
//icp
#include <pcl/registration/icp.h>

#include <set>
#include <map>

//opencv
#include <opencv2/features2d.hpp> //fastfeature detector
#include <opencv2/xfeatures2d/nonfree.hpp> //surf
#include "opencv2/ml.hpp"

//imsim
#include <image_similarity/types/similarity_types.h>
#include <pcl/features/rift.h>
#include <pcl/features/intensity_gradient.h>


//colorhist
#include <robosherlock/compare.h>

using namespace uima;
using namespace cv;
using namespace cv::xfeatures2d;


using std::cout;
using std::vector;


class imsim : public DrawingAnnotator {
private:
    typedef pcl::PointXYZRGBA PointT;

    std::string classificationSpawnedInUE ="";

    cv::Mat maskone, masktwo; //test for masks
    // This identifier is used to reference the CAS of another AAE that ran before
    // *this* AAE. It is access via rs::CASConsumerContext
    std::string other_cas_id;

    cv::Mat color, object_, rgb_, depth_, other_cas_img_, other_cas_depth_, img_matches, img_matches_test;
    pcl::PointCloud<PointT>::Ptr bs_cloud_;
    pcl::PointCloud<PointT>::Ptr other_cas_cloud_;

    std::map<std::string, vector<cv::KeyPoint>> keypoints_bs_compl, keypoints_rw_compl;
    std::map<std::string, vector<cv::DMatch>> complete_image_matches;

    std::set<std::string> list_of_object_classifications_surf, list_of_object_classifications_pcl;
    std::map<std::string, cv::Rect> bs_roi_map, rw_roi_map;
    std::map<std::string, cv::Mat> bs_rois_with_mask;
    std::map<std::string,vector<rs::ColorHistogram>> rw_hist_map, bs_hist_map;

    std::map<std::string, std::string> id_to_classname_;
    std::list<std::string> seenClassifications;
    std::string currentClassification = "";

//pcl params
    std::map<std::string, pcl::PointCloud<PointT>::Ptr> bs_pcl_map, rw_pcl_map;
    std::map<std::string, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> bs_keypoint_map_3d, rw_keypoint_map_3d;
    std::map<std::string, pcl::Correspondences> corr_per_object;

    pcl::PointCloud<pcl::Normal>::Ptr bs_norm, rw_norm;

    pcl::PointCloud<PointT>::Ptr dispCloud;



    int iteration = -1;

    bool use_hd_images_ = true;
    bool save_images_for_debug = true;
    bool nextMatch = false;
    bool firstRun = true; //to have a default image in the visualizer

    enum {
        ALL_TOGETHER,
        SINGLE_MATCHES,
        ALL_TOGETHER_CLOUD,
        BS_CLOUD_ONLY,
        OTHER_CAS_CLOUD_ONLY,
        KEYPOINTS_CORRESPONDENCES_SINGLE_OBJECT,
        KEYPOINTS_CORRESPONDENCES_SINGLE_OBJECT_WITH_OBJECT,
        COLOR_HIST,
        MASKONE,
        MASKTWO
    } dispMode;


    //colorhist params
    // ClusterColor C&P
    enum COLORS {
        RED = 0,
        YELLOW,
        GREEN,
        CYAN,
        BLUE,
        MAGENTA,
        WHITE,
        BLACK,
        GREY,
        COUNT
    };
    int min_value_color_;
    int min_saturation_color_;
    int max_value_black_;
    int min_value_white_;
    int histogram_cols_;
    int histogram_rows_;

    const double color_range_;

    std::vector<int> color_positions_;

    std::vector<cv::Scalar> colors_;

    std::vector<std::string> color_names_;
    std::vector<cv::Rect> cluster_rois_;
    std::vector<std::vector<int>> color_ids_;
    std::vector<std::vector<float>> color_ratios_;

    int camera_id_ = 0;
    ros::NodeHandle nh_;
    bool semantic_label_;


    //RandomForest + Csvread /write
    std::ofstream myFile;
    int rounds = 0;
    int min_samples_split;
    int max_depth;
    int max_features;
    std::string training_data_path;
    std::string classify_data_path;
    std::string ground_truth_path;
    cv::Ptr<cv::ml::RTrees> rndTree;
    cv::Ptr<cv::ml::LogisticRegression> logReg;
    bool useRTree = true; //used to flip between randomforest (rtree) and logistic regression in an easy way.
    float logRegThreshold; //threshold which defines how sure the logisticregression needs to be to override the knn result.

    vector<std::string> alreadyWrittenToCsv;
public:

    imsim() : DrawingAnnotator(__func__), nh_("~"), min_value_color_(60),
              min_saturation_color_(60), max_value_black_(60), min_value_white_(120),
              histogram_cols_(16),
              histogram_rows_(16), color_range_(256.0 / 6.0), semantic_label_(false) {
        color_positions_.resize(6);
        for (size_t i = 0; i < 6; ++i) {
            color_positions_[i] = (int) (i * color_range_ + color_range_ / 2.0 + 0.5);
        }

        color_names_.resize(COUNT);
        color_names_[RED] = "red";
        color_names_[YELLOW] = "yellow";
        color_names_[GREEN] = "green";
        color_names_[CYAN] = "cyan";
        color_names_[BLUE] = "blue";
        color_names_[MAGENTA] = "magenta";
        color_names_[WHITE] = "white";
        color_names_[BLACK] = "black";
        color_names_[GREY] = "grey";

        colors_.resize(COUNT);
        colors_[RED] = CV_RGB(255, 0, 0);
        colors_[YELLOW] = CV_RGB(255, 255, 0);
        colors_[GREEN] = CV_RGB(0, 255, 0);
        colors_[CYAN] = CV_RGB(0, 255, 255);
        colors_[BLUE] = CV_RGB(0, 0, 255);
        colors_[MAGENTA] = CV_RGB(255, 0, 255);
        colors_[WHITE] = CV_RGB(255, 255, 255);
        colors_[BLACK] = CV_RGB(0, 0, 0);
        colors_[GREY] = CV_RGB(127, 127, 127);
    }

    //stuff for colorhistogram taken from old colorhiststate of Patrick Mania
    TyErrorId initialize(AnnotatorContext &ctx) {
        outInfo("initialize");
        cout << "OpenCV Version used:" << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << endl;
        if (ctx.isParameterDefined("otherCASId")) {
            ctx.extractValue("otherCASId", other_cas_id);
            outInfo("Using AAE/CAS identified by '" << other_cas_id << "' for imagesimilarity");
        }
        if (ctx.isParameterDefined("camera_id")) {
            ctx.extractValue("camera_id", camera_id_);
            outInfo("Using camera_id '" << camera_id_ << "'");
        }
        if (ctx.isParameterDefined("use_hd_images")) {
            ctx.extractValue("use_hd_images", use_hd_images_);
            outInfo("Use HD Image streams from Main Cam and Belief State? " << use_hd_images_);
        }
        if (ctx.isParameterDefined("minValueColor")) {
            ctx.extractValue("minValueColor", min_value_color_);
        }
        if (ctx.isParameterDefined("minSaturationColor")) {
            ctx.extractValue("minSaturationColor", min_saturation_color_);
        }
        if (ctx.isParameterDefined("maxValueBlack")) {
            ctx.extractValue("maxValueBlack", max_value_black_);
        }
        if (ctx.isParameterDefined("minValueWhite")) {
            ctx.extractValue("minValueWhite", min_value_white_);
        }
        if (ctx.isParameterDefined("histogramCols")) {
            ctx.extractValue("histogramCols", histogram_cols_);
        }
        if (ctx.isParameterDefined("histogramRows")) {
            ctx.extractValue("histogramRows", histogram_rows_);
        }
        if (ctx.isParameterDefined("semantic_label")) {
            ctx.extractValue("semantic_label", semantic_label_);
        }
        if (ctx.isParameterDefined("min_samples_split")) {
            ctx.extractValue("min_samples_split", min_samples_split);
            outInfo("Using min_samples_split '" << min_samples_split << "' for imagesimilarity");
        }
        if (ctx.isParameterDefined("max_depth")) {
            ctx.extractValue("max_depth", max_depth);
            outInfo("Using max_depth '" << max_depth << "' for imagesimilarity");
        }
        if (ctx.isParameterDefined("max_features")) {
            ctx.extractValue("max_features", max_features);
            outInfo("Using max_features '" << max_features << "' for imagesimilarity");
        }
        if (ctx.isParameterDefined("training_data")) {
            ctx.extractValue("training_data", training_data_path);
            outInfo("Using training_data '" << training_data_path << "' for imagesimilarity");
        }
        if (ctx.isParameterDefined("classify_data")) {
            ctx.extractValue("classify_data", classify_data_path);
            outInfo("Using classify_data '" << classify_data_path << "' for imagesimilarity");
        }
        if (ctx.isParameterDefined("ground_truth")) {
            ctx.extractValue("ground_truth", ground_truth_path);
            outInfo("Using ground_truth '" << ground_truth_path << "' for imagesimilarity");
        }
        if (ctx.isParameterDefined("useRTree")) {
            ctx.extractValue("useRTree", useRTree);
            outInfo("If true using RandomForest for imagesimilarity else logistic regression: '" << useRTree);
        }
        if (ctx.isParameterDefined("logRegThreshold")) {
            ctx.extractValue("logRegThreshold", logRegThreshold);
            outInfo("Using " << logRegThreshold << " as decision threshold to use logistic regression in addition to knn objectclasssifier.");
        }

        //write header for csv file
        myFile.open("/home/ros/Desktop/similarity.csv", std::ofstream::out | std::ofstream::app);
        myFile << "Classification" << "," << "Comparedto" << "," << "identical" << "," << "SurfMatches" << ","
               << "PCLMatches" << "," << "ICPSimilarityscore" << "," << "ColorHistDistance" << "," << "guess" << "," << "confidence" << ","
               << "mockedPipeline" << ","
               << "neighborsWithDist" << std::endl;
        myFile.close();

        loadTrainingDataAndTrain();
        validateTraining();

        return UIMA_ERR_NONE;
    }

    TyErrorId destroy() {
        outInfo("destroy");
        return UIMA_ERR_NONE;
    }

    //searches the keypoints in the smaller image snippets which are later used to calculate the image similarity in the complete image. with this way there are
    // only keypoints detected at the objects we are interested in. if save_images_for_debug is true it will save the snippet images and an image with the matches
    void surfWithDebugImages(Ptr<SURF> surf, BFMatcher matcher, CAS &tcas, cv::Mat bs_cluster, cv::Mat rw_cluster,
                             std::string classification) {
        //setup for image similarity via surf

        iteration++;
        /* iteration++;
         for (std::string classification : list_of_object_classifications_surf) {
             //if (classification != "unknown") { //we are only interested in classified stuff

             cv::Rect bs_roi = bs_roi_map[classification];
             cv::Rect rw_roi = rw_roi_map[classification];

             if (bs_roi.empty() || rw_roi.empty()) {
                 outInfo("bs or rw roi empty");
                 continue;
             }

             //cut interesting part out of the full size image
             cv::Mat bs_cluster = rgb_(bs_roi);
             cv::Mat rw_cluster = other_cas_img_(rw_roi);*/

        //set save_images_for_debug for debug reasons or if you are interested in the snippet images and matching between snippet images
        imwrite("/home/ros/Desktop/Aufnahmen/JPG/bs_" + std::to_string(iteration) + "_" + classification +
                "_cluster.jpg", bs_cluster);
        imwrite("/home/ros/Desktop/Aufnahmen/JPG/rw_" + std::to_string(iteration) + "_" + classification +
                "_cluster.jpg", rw_cluster);

        imwrite("/home/ros/Desktop/Aufnahmen/Pipeline/JPG/bs_" + id_to_classname_[classification] +
                ".jpg", bs_cluster);
        imwrite("/home/ros/Desktop/Aufnahmen/Pipeline/JPG/rw_" + id_to_classname_[classification] +
                ".jpg", rw_cluster);

        vector<cv::KeyPoint> keypoints0, keypoints1;

        //detect keypoints
        surf->detect(bs_cluster, keypoints0);
        surf->detect(rw_cluster, keypoints1);

        if (keypoints0.size() == 0) {
            outInfo("bs keypoints size 0");
        }

        if (keypoints1.size() == 0) {
            outInfo("rw keypoints size 0");
        }

        // std::map<std::string, vector<cv::KeyPoint>> bs_keypoint_map, rw_keypoint_map;
        //for complete image drawing
        //   bs_keypoint_map[classification] = keypoints0;
        //   rw_keypoint_map[classification] = keypoints1;


        // this code here is only needed to draw the matches on the imagesnippets corresponding to each classification.
        // if the result is not saved there is not point in calculating everything for it

        Mat descriptors0, descriptors1;
        // computing descriptors
        surf->compute(bs_cluster, keypoints0, descriptors0);
        surf->compute(rw_cluster, keypoints1, descriptors1);


        // matching descriptors
        vector<DMatch> matches;
        matcher.match(descriptors0, descriptors1, matches);

        vector<Point2f> bs_points, rw_points;

        //get all keypoints which are part of a match
        for (size_t i = 0; i < matches.size(); i++) {
            bs_points.push_back(keypoints0[matches[i].queryIdx].pt);
            rw_points.push_back(keypoints1[matches[i].trainIdx].pt);
        }
        cv::Mat mask;
        outInfo("bspointssize " << bs_points.size());
        outInfo("rwpointssize " << rw_points.size());
        cv::findHomography(bs_points, rw_points, cv::RANSAC, 5.0, mask);
        //fill the homography mask which is used to filter the keypoints / matches. we identifiy inliers and outliers and use only the inliers

        vector<cv::KeyPoint> keypoints_bs_after_homography, keypoints_rw_after_homography;
        for (int i = 0; i < mask.rows; i++) {
            if ((unsigned int) mask.at<uchar>(i) == 1) { //ask.at<uchar>(i) == 1 = inlier. 0 would be outlier
                keypoints_bs_after_homography.push_back(keypoints0[matches[i].queryIdx]);
                keypoints_rw_after_homography.push_back(keypoints1[matches[i].trainIdx]);
            }
        }
        //calculate new descriptors on remaining keypoints
        surf->compute(bs_cluster, keypoints_bs_after_homography, descriptors0);
        surf->compute(rw_cluster, keypoints_rw_after_homography, descriptors1);


        // matching descriptors
        matcher.match(descriptors0, descriptors1, matches);

        cv::drawMatches(bs_cluster, keypoints0, rw_cluster, keypoints1, matches, img_matches_test,
                        Scalar::all(-1),
                        Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        imwrite("/home/ros/Desktop/Aufnahmen/JPG/matches_" + std::to_string(iteration) + "_" +
                classification +
                ".jpg",
                img_matches_test);
        //}
    }

    void surf(Ptr<SURF> surf, BFMatcher matcher, CAS &tcas, cv::Mat bs_cluster, cv::Mat rw_cluster,
              std::string classification, bool withVisualiser, std::string mockClassification) {
        complete_image_matches.clear();
        keypoints_bs_compl.clear();
        keypoints_rw_compl.clear();


        /*for (std::string classification : list_of_object_classifications_surf) {
            if (classification != "unknown") {*/



        /*if (bs_roi.empty() || rw_roi.empty()) {
            outInfo("bs or rw roi empty");
            continue;
        }

        //cut interesting part out of the full size image
       cv::Mat bs_cluster = rgb_(bs_roi);
        cv::Mat rw_cluster = other_cas_img_(rw_roi);*/



        vector<cv::KeyPoint> bs_kp, rw_kp;
        //detect keypoints
        surf->detect(bs_cluster, bs_kp);
        surf->detect(rw_cluster, rw_kp);


        /* vector<KeyPoint> bs_kp, rw_kp;
         bs_kp = bs_keypoint_map[classification]; //get image snippets for each classification
         rw_kp = rw_keypoint_map[classification];*/



        //  cv::Rect bs_roi = bs_roi_map[classification]; // get rois for each classification
        //cv::Rect rw_roi = rw_roi_map[classification];

        //add the roi coordinates to the keypoint coordinates. this is needed because we calculated the keypoints on the smaller image snippets to remove
        // unwanted background stuff and get better keypoints. now we put these found keypoints back into the original images to create a nice image with matches and keypoints
        if (withVisualiser) {
            cv::Rect bs_roi = bs_roi_map[classification];
            cv::Rect rw_roi = rw_roi_map[classification];
            if (bs_kp.size() > 0) {
                for (int i = 0; i < bs_kp.size(); i++) {
                    bs_kp[i].pt.x += bs_roi.x;
                    bs_kp[i].pt.y += bs_roi.y;
                }
                keypoints_bs_compl[classification] = bs_kp;
            } else {
                outInfo("bs keypoints empty");
            }

            if (rw_kp.size() > 0) {
                for (int i = 0; i < rw_kp.size(); i++) {
                    rw_kp[i].pt.x += rw_roi.x;
                    rw_kp[i].pt.y += rw_roi.y;
                }
                keypoints_rw_compl[classification] = rw_kp;
            } else {
                outInfo("rw keypoints empty");
            }
        }

        Mat descriptors_bs, descriptors_rw;
        //get descriptors
        surf->compute(rgb_, bs_kp, descriptors_bs);
        surf->compute(other_cas_img_, rw_kp, descriptors_rw);

        vector<DMatch> matches_before_homography;
        //get matches the first time. these will be filtered via findHomography
        matcher.match(descriptors_bs, descriptors_rw, matches_before_homography);

        vector<Point2f> bs_points, rw_points;

        //get all keypoints which are matched to another and use them for the homography
        for (size_t i = 0; i < matches_before_homography.size(); i++) {
            bs_points.push_back(bs_kp[matches_before_homography[i].queryIdx].pt);
            rw_points.push_back(rw_kp[matches_before_homography[i].trainIdx].pt);
        }
        cv::Mat mask;

        outInfo("bspointssize " << bs_points.size());
        outInfo("rwpointssize " << rw_points.size());
        if(bs_points.size() == 0 || rw_points.size() == 0)
        {
            outInfo("hier");
        }
        cv::findHomography(bs_points, rw_points, cv::RANSAC, 5.0,
                           mask); //use findhomography to get the mask. now we are able to identify keypoints which are inside the object and which are not


        vector<cv::KeyPoint> keypoints_bs_after_homography, keypoints_rw_after_homography;
        //filter all found keypoints with the homography mask => filter outliers out and leave only inliers. If mask.at<uchar>(i) == 1 it is an inlier
        for (int i = 0; i < mask.rows; i++) {
            if ((unsigned int) mask.at<uchar>(i) == 1) {
                keypoints_bs_after_homography.push_back(
                        bs_kp[matches_before_homography[i].queryIdx]); //get the keypoint at the found maskposition
                keypoints_rw_after_homography.push_back(rw_kp[matches_before_homography[i].trainIdx]);
            }
        }

        Mat descriptors0, descriptors1;
        //extract descriptors for matching
        surf->compute(rgb_, keypoints_bs_after_homography, descriptors0);
        surf->compute(other_cas_img_, keypoints_rw_after_homography, descriptors1);


        //matching descriptors
        vector<DMatch> matches;
        matcher.match(descriptors0, descriptors1, matches);
        complete_image_matches[classification] = matches; // save matches for the view later


        rs::SceneCas cas(tcas);
        rs::Scene scene = cas.getScene();

        std::string classNameToSave = id_to_classname_[classification];
        outInfo(classNameToSave);
        if(mockClassification != "") {
            classNameToSave = mockClassification;
        }
        outInfo(classNameToSave);
        //save surf result to cas and create new object in cas if none with the classification name exists
        std::vector<image_similarity::SimilarityObject> imsimobjects;
        scene.annotations.filter(imsimobjects);
        bool contained = false;
        for (image_similarity::SimilarityObject iso: imsimobjects) {
             if (iso.classification.get() == classNameToSave) {
                iso.matches_surf(matches.size());
                contained = true;
                break;
            }
        }
        if (contained == false) {
            image_similarity::SimilarityObject imsimObject = rs::create<image_similarity::SimilarityObject>(
                    tcas);
            imsimObject.classification.set(classNameToSave);
            imsimObject.matches_surf(matches.size());
            scene.annotations.append(imsimObject);
        }

        outInfo("amount of matches in complete image " << complete_image_matches.size());
        //    }
        //   }
    }

    // main process where everything starts
    TyErrorId processWithLock(CAS &tcas, ResultSpecification const &res_spec) {
        //clear lists
        id_to_classname_.clear();
        alreadyWrittenToCsv.clear();

        outInfo("process start");

        rs::StopWatch clock;

        getDataFromBothCasAndPrepareForUse(tcas);

        //use surf / create surf + bf matcher
        Ptr<SURF> surf_ptr = SURF::create(500);
        surf_ptr->setExtended(true);
        BFMatcher matcher(NORM_L1);
        if (!bs_roi_map.empty()) { //as long as we have a roi in the beliefstate
            iteration++;
            for (std::string classification : list_of_object_classifications_surf) {

                //if (classification != "unknown") { //we are only interested in classified stuff

                classificationSpawnedInUE = id_to_classname_[classification];
                outInfo("classificationSpawnedInUE= " << classificationSpawnedInUE);
                cv::Rect bs_roi = bs_roi_map[classification];
                cv::Rect rw_roi = rw_roi_map[classification];

                if (bs_roi.empty() || rw_roi.empty()) {
                    outInfo("bs or rw roi empty");
                    continue;
                }

                //cut interesting part out of the full size image
                cv::Mat bs_cluster = rgb_(bs_roi);
                cv::Mat rw_cluster = other_cas_img_(rw_roi);

                //once run surf with image output
                surfWithDebugImages(surf_ptr, matcher, tcas, bs_cluster, rw_cluster, classification);
                //run surf without images output
                surf(surf_ptr, matcher, tcas, bs_cluster, rw_cluster, classification, true,"");

                //get colorhistograms
                vector<rs::ColorHistogram> bs_hist = bs_hist_map[classification];
                vector<rs::ColorHistogram> rw_hist = rw_hist_map[classification];

                //compare them
                colorHist_new(tcas, bs_hist, rw_hist, classification, "");
            }
        } else {
            outInfo("bs roi map is empty");
        }

        if (!bs_pcl_map.empty()) {
            for (std::string classification : list_of_object_classifications_pcl) {


                pcl::PointCloud<PointT>::Ptr bs_cloud = bs_pcl_map[classification];
                pcl::PointCloud<PointT>::Ptr rw_cloud = rw_pcl_map[classification];

                if (bs_cloud->points.size() == 0 || rw_cloud->points.size() == 0) {
                    outInfo("Pcl with size 0");
                    continue;
                }
                //runicp
                 icp_pipeline(tcas, bs_cloud, rw_cloud, classification,"");
                //run pcl
                pcl_pipeline(tcas, bs_cloud, rw_cloud, classification,"");
            }

        } else {
            outInfo("bs pcl map is empty");
        }

       rs::SceneCas cas(tcas);
       rs::Scene scene = cas.getScene();

       std::vector<rs_bs::BeliefStateObject> clusters;
        scene.identifiables.filter(clusters);

        cluster_rois_.resize(clusters.size());
        color_ids_.resize(clusters.size(), std::vector<int>(COUNT));
        color_ratios_.resize(clusters.size(), std::vector<float>(COUNT));
        for (size_t idx = 0; idx < clusters.size(); ++idx) {

            //    std::string classification = id_to_classname_[clusters[idx].rsObjectId.get()];
            //  rs::ImageROI image_rois = clusters[idx].rois.get();

            cv::Rect roi = bs_roi_map[clusters[idx].rsObjectId.get()];
            cv::Mat mask = bs_rois_with_mask[clusters[idx].rsObjectId.get()];

            if(save_images_for_debug) {
                outInfo("mask " << mask.size << " " << mask.depth()<< " " << mask.type());
                FileStorage fs("/home/ros/Desktop/Aufnahmen/Pipeline/JPG/roi_" +
                               id_to_classname_[clusters[idx].rsObjectId.get()] +
                               ".yml", FileStorage::WRITE);
                if (fs.isOpened()) {
                    fs << "x" << roi.x << "y" << roi.y;
                    fs << "width" << roi.width << "height" << roi.height;
                    fs.release();
                }
                maskone = mask;
                imwrite("/home/ros/Desktop/Aufnahmen/Pipeline/JPG/mask_" +
                        id_to_classname_[clusters[idx].rsObjectId.get()] +
                        ".jpg", mask);
            }
            //colorhist(tcas, roi, mask, clusters, idx, scene, "");

        }



//run random forest on cas data. also mock "rerun" if random forest result is objects where not the smae
        runClassifierOnLiveDataAndWriteCsv(tcas);

        outInfo("took: " << clock.getTime() << " ms.");
        return UIMA_ERR_NONE;

    }

    void getDataFromBothCasAndPrepareForUse(CAS &tcas) {
        rs::SceneCas cas(tcas);
        iteration++;

        bs_cloud_ = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
        other_cas_cloud_ = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);

        pcl::PointCloud<PointT>::Ptr new_display_cloud(new pcl::PointCloud<PointT>());
        pcl::PointCloud<pcl::Normal>::Ptr new_bs_norm(new pcl::PointCloud<pcl::Normal>);
        pcl::PointCloud<pcl::Normal>::Ptr new_rw_norm(new pcl::PointCloud<pcl::Normal>);
        bs_norm = new_bs_norm;
        rw_norm = new_rw_norm;
        dispCloud = new_display_cloud;

        // set everything for later / get object image, depth image , colorimage
        outInfo(" use hd for imagesimilarity: " + use_hd_images_);
        outInfo(use_hd_images_);
        if (use_hd_images_) {
            cas.get(VIEW_OBJECT_IMAGE_HD, object_);
            cas.get(VIEW_COLOR_IMAGE_HD, rgb_);
            cas.get(VIEW_DEPTH_IMAGE_HD, depth_);
        } else {
            cas.get(VIEW_OBJECT_IMAGE, object_);
            cas.get(VIEW_COLOR_IMAGE, rgb_);
            cas.get(VIEW_DEPTH_IMAGE, depth_);
        }
        cas.get(VIEW_CLOUD, *bs_cloud_);


        rs::Scene scene = cas.getScene();

        uima::CAS *other_cas;
        other_cas = rs::CASConsumerContext::getInstance().getCAS(other_cas_id);
        if (!other_cas) {
            outError("Couldn't fetch CAS identified by '"
                             << other_cas_id
                             << "'. Make sure you have loaded an AAE with that name and "
                             << " that you've set 'otherCASId' in this config");
        }

        rs::SceneCas other_cas_scene(*other_cas);
        //get color image / depthimage
        if (use_hd_images_) {
            other_cas_scene.get(VIEW_COLOR_IMAGE_HD, other_cas_img_);
            other_cas_scene.get(VIEW_DEPTH_IMAGE_HD, other_cas_depth_);
        } else {
            other_cas_scene.get(VIEW_COLOR_IMAGE, other_cas_img_);
            other_cas_scene.get(VIEW_DEPTH_IMAGE, other_cas_depth_);
        }
        other_cas_scene.get(VIEW_CLOUD, *other_cas_cloud_);

        if (bs_cloud_->empty() && other_cas_cloud_->empty()) {
            outError("Clouds empty. Skipping");
        }


        if (rgb_.cols != other_cas_img_.cols || rgb_.rows != other_cas_img_.rows) {
            outError("BS and RW RGB image resolution Mismatch! "
                     "Resolutions of the other cas cam (" << other_cas_img_.cols << "x" << other_cas_img_.rows <<
                                                          ") and the synthetic views (" << rgb_.cols << "x" << rgb_.rows
                                                          << ") differ");
        }

        //prepare for run
        prepare_surf_and_colorhist(cas, other_cas_scene);
        prepare_icpAndPcl(cas, other_cas_scene);

    }

    //gets roi for object and saves it from beliefstate and real camera
    //also gets the colorhist for both and saves it
    void prepare_surf_and_colorhist(rs::SceneCas cas, rs::SceneCas other_cas_scene) {
        //from here rw stuff
        std::vector<rs::Object> rwobjects;
        other_cas_scene.get(VIEW_OBJECTS, rwobjects);
        outInfo("Found " << rwobjects.size() << " rs::Object in RW CAS");
        for (auto rwo: rwobjects) {
            cv::Rect rw_roi;
            if (use_hd_images_) {
                rs::conversion::from(rwo.rois().roi_hires(), rw_roi);
            } else {
                rs::conversion::from(rwo.rois().roi(), rw_roi);
            }
            rw_roi_map[rwo.id.get()] = rw_roi; //save roi

            std::vector<rs::ColorHistogram> colors;
            rwo.annotations.filter(colors);
            rw_hist_map[rwo.id.get()] = colors; //get colorhistogram and save it


            getRwoClassificationPrepared(rwo);
        }

        //from here bso stuff
        std::vector<rs_bs::BeliefStateObject> bsobjects;
        cas.get(VIEW_ALL_BS_OBJECTS, bsobjects); //get all bs objects
        //scene.identifiables.filter(bsobjects);
        outInfo("Found " << bsobjects.size() << " rs::BeliefStateObject in BS CAS");
        for (auto bso: bsobjects) {
            cv::Rect bs_roi;
            cv::Mat bs_mask;
            if (use_hd_images_) {
                rs::conversion::from(bso.rois().roi_hires(), bs_roi);
                rs::conversion::from(bso.rois().mask_hires(), bs_mask);
            } else {
                rs::conversion::from(bso.rois().roi(), bs_roi);
                rs::conversion::from(bso.rois().mask(), bs_mask);
            }
            // Fetch the class name from the matching RW cluster
         //   outInfo("roi area  " << bs_roi.x << " " << bs_roi.y << " " << bs_roi.height << " " << bs_roi.width
           //                      << "  simulationactorid  " << bso.simulationActorId.get());

            //get classnames from list of objectclassifications with the bso object id
            if (std::find(list_of_object_classifications_surf.begin(), list_of_object_classifications_surf.end(),
                          bso.rsObjectId.get()) != list_of_object_classifications_surf.end()) {
                std::string class_name = id_to_classname_[bso.rsObjectId.get()];
                outInfo(bso.rois().roi().size);
                outInfo(bs_roi.size());
                outInfo(bs_mask.size);
                //bs_mask = bs_mask.t(); //mask as strangly the wrong rows / cols for the roi. so here we flip it.
                outInfo(bs_mask.size);
                bs_roi_map[bso.rsObjectId.get()] = bs_roi;//save bsroi

                std::vector<rs::ColorHistogram> colors;
                bso.annotations.filter(colors);
                bs_hist_map[bso.rsObjectId.get()] = colors; //get colorhistogram and save it

                bs_rois_with_mask[bso.rsObjectId.get()] = bs_mask;  //maskinput for colorhiststuff

            } else {
                outInfo("bso with rsobjectid = " << bso.rsObjectId.get() << " has no match with same id.");
            }

        }
    }

    //prepares for icp and pcl so that no more cas loading is needed
    void prepare_icpAndPcl(rs::SceneCas cas, rs::SceneCas other_cas_scene) {

        //get recognized objects
        //from here rw stuff
        std::vector<rs::Object> rwobjects;
        other_cas_scene.get(VIEW_OBJECTS, rwobjects);
        outInfo("Found " << rwobjects.size() << " rs::Object in RW CAS");

        for (auto rwo: rwobjects) {

            getRwoClassificationPrepared(rwo);

            //Get point cloud of cluster
            pcl::PointIndicesPtr indices(new pcl::PointIndices());
            rs::conversion::from(static_cast<rs::ReferenceClusterPoints>(rwo.points.get()).indices.get(), *indices);

            pcl::PointCloud<PointT>::Ptr rw_cluster_cloud(new pcl::PointCloud<PointT>);

            //get indices from cloud
            pcl::ExtractIndices<PointT> eirw;
            eirw.setInputCloud(other_cas_cloud_);
            eirw.setIndices(indices);
            eirw.filter(*rw_cluster_cloud);
            std::cerr << "RWPointCloud complete: " << other_cas_cloud_->width * other_cas_cloud_->height
                      << " data points." << std::endl;
            std::cerr << "RWPointCloud representing the planar component: "
                      << rw_cluster_cloud->width * rw_cluster_cloud->height << " data points." << std::endl;

            rw_pcl_map[rwo.id.get()] = rw_cluster_cloud;
            pcl::copyPointCloud(*rw_cluster_cloud, *dispCloud);

        }


        //from here bso stuff
        std::vector<rs_bs::BeliefStateObject> bsobjects;
        cas.getScene().identifiables.filter(bsobjects);
        outInfo("Found " << bsobjects.size() << " rs::BeliefStateObject in BS CAS");

        for (auto bso: bsobjects) {

            //get pointcloud of cluster
            pcl::PointIndicesPtr indices(new pcl::PointIndices());
            rs::conversion::from(static_cast<rs::ReferenceClusterPoints>(bso.points.get()).indices.get(), *indices);

            //https://pointclouds.org/documentation/tutorials/extract_indices.html
            pcl::PointCloud<PointT>::Ptr bs_cluster_cloud(new pcl::PointCloud<PointT>);

            pcl::ExtractIndices<PointT> eibs;
            eibs.setInputCloud(bs_cloud_);
            eibs.setIndices(indices);
            eibs.filter(*bs_cluster_cloud);
            std::cerr << "BSPointCloud complete: " << bs_cloud_->width * bs_cloud_->height << " data points."
                      << std::endl;
            std::cerr << "BSPointCloud representing the planar component: "
                      << bs_cluster_cloud->width * bs_cluster_cloud->height << " data points." << std::endl;

            if (std::find(list_of_object_classifications_pcl.begin(), list_of_object_classifications_pcl.end(),
                          bso.rsObjectId.get()) != list_of_object_classifications_pcl.end()) {
                // Fetch the class name from the matching RW cluster
                std::string class_name = id_to_classname_[bso.rsObjectId.get()];
                //list_of_object_classifications.insert(bso.rsObjectId.get());
                bs_pcl_map[bso.rsObjectId.get()] = bs_cluster_cloud;
                *dispCloud += *bs_cluster_cloud;
            } else {
                outInfo("bso with rsobjectid = " << bso.rsObjectId.get() << " has no match with same id.");
            }
        }
    }

    //gets the classifications from the rsknn / baseline classifier and saves them. each classification is saved once
    //creates list of objectclassifications and maps the objectid to the classification
    void getRwoClassificationPrepared(rs::Object rwo) {
        std::string class_name;
        std::vector<rs::Classification> classes;
        rwo.annotations.filter(classes);
        outInfo("Found " << classes.size() << " object annotations in rwo");

        if (classes.size() == 0) {
            outInfo("No classification information for cluster ");
            class_name = "unknown";
        } else {
            class_name = classes[0].classname.get();
            outInfo("rwo classname is " << class_name << " with rwoid : " << rwo.id.get());
        }

        //look if already in the list if not add
        if (std::find(list_of_object_classifications_surf.begin(), list_of_object_classifications_surf.end(),
                      rwo.id.get()) != list_of_object_classifications_surf.end()) {
            list_of_object_classifications_surf.insert(rwo.id.get());
            id_to_classname_[rwo.id.get()] = class_name;
        } else if (list_of_object_classifications_surf.empty()) {
            list_of_object_classifications_surf.insert(rwo.id.get());
            id_to_classname_[rwo.id.get()] = class_name;
        }

        //look if already in the list if not add
        if (std::find(list_of_object_classifications_pcl.begin(), list_of_object_classifications_pcl.end(),
                      rwo.id.get()) != list_of_object_classifications_pcl.end()) {
            list_of_object_classifications_pcl.insert(rwo.id.get());
            id_to_classname_[rwo.id.get()] = class_name;
        } else if (list_of_object_classifications_pcl.empty()) {
            list_of_object_classifications_pcl.insert(rwo.id.get());
            id_to_classname_[rwo.id.get()] = class_name;
        }
    }

    void icp_pipeline(CAS &tcas, pcl::PointCloud<PointT>::Ptr bs_cloud, pcl::PointCloud<PointT>::Ptr rw_cloud,
                      std::string classification, std::string mockClassification) {

        /* for (std::string classification : list_of_object_classifications_pcl) {
             if (classification != "unknown") {

                 pcl::PointCloud<PointT>::Ptr bs_cloud = bs_pcl_map[classification];
                 pcl::PointCloud<PointT>::Ptr rw_cloud = rw_pcl_map[classification];

                 if (bs_cloud->points.size() == 0 || rw_cloud->points.size() == 0) {
                     outInfo("Pcl with size 0");
                     continue;
                 }*/
//prepeare icp with parameters
        pcl::IterativeClosestPoint<PointT, PointT> icp;
        icp.setInputSource(bs_cloud);
        icp.setInputTarget(rw_cloud);
        icp.setMaximumIterations(250);
        icp.setEuclideanFitnessEpsilon(1e-4);

        //run icp
        pcl::PointCloud<PointT>::Ptr this_cas_cluster_cloud_aligned(new pcl::PointCloud<PointT>());
        icp.align(*this_cas_cluster_cloud_aligned);

        outInfo("ICP done. Has converged? " << icp.hasConverged()
                                            << " score: " << icp.getFitnessScore());
        //outInfo("ICP found transform: " << icp.getFinalTransformation());


        rs::SceneCas cas(tcas);
        rs::Scene scene = cas.getScene();
//save fitnessscore in cas
        std::string classNameToSave = id_to_classname_[classification];
        if(mockClassification != "") {
            classNameToSave = mockClassification;
        }

        std::vector<image_similarity::SimilarityObject> imsimobjects;
        scene.annotations.filter(imsimobjects);
        bool contained = false;
        for (image_similarity::SimilarityObject iso: imsimobjects) {
            if (iso.classification.get() == classNameToSave) {
                iso.similarityscore_icp(icp.getFitnessScore());
                contained = true;
                break;
            }
        }
        if (contained == false) {
            image_similarity::SimilarityObject imsimObject = rs::create<image_similarity::SimilarityObject>(
                    tcas);
            imsimObject.classification.set(classNameToSave);
            imsimObject.similarityscore_icp(icp.getFitnessScore());
            scene.annotations.append(imsimObject);
        }
    }

    //see thesis for diagram / run description
    //in short uses sift to find keypoints, mapping via fpfh, rejection with distance and ransac then save to cas
    void pcl_pipeline(CAS &tcas, pcl::PointCloud<PointT>::Ptr bs_cloud, pcl::PointCloud<PointT>::Ptr rw_cloud,
                      std::string classification, std::string  mockClassification) {

        /* for (std::string classification : list_of_object_classifications_pcl) {
             if (classification != "unknown") {

                 pcl::PointCloud<PointT>::Ptr bs_cloud = bs_pcl_map[classification];
                 pcl::PointCloud<PointT>::Ptr rw_cloud = rw_pcl_map[classification];*/

        if (save_images_for_debug) {
            pcl::io::savePCDFileBinary(
                    "/home/ros/Desktop/Aufnahmen/Pipeline/PCL/rw_" + id_to_classname_[classification] +
                    ".pcd", *rw_cloud);
            pcl::io::savePCDFileBinary(
                    "/home/ros/Desktop/Aufnahmen/Pipeline/PCL/bs_" + id_to_classname_[classification] +
                    ".pcd", *bs_cloud);
        }

        /*if (bs_cloud->points.size() == 0 || rw_cloud->points.size() == 0) {
            outInfo("Pcl with size 0");
            continue;
        }*/

        // https://github.com/PointCloudLibrary/pcl/blob/master/examples/keypoints/example_sift_keypoint_estimation.cpp
        pcl::SIFTKeypoint<pcl::PointXYZRGBA, pcl::PointWithScale> sift;
        pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGBA>());
        sift.setSearchMethod(kdtree);
        //sift.setScales(0.003, 8, 8);
//sift.setScales(0.003, 4, 2);

        sift.setScales(0.003, 4, 4);
        sift.setMinimumContrast(4);


        pcl::PointCloud<pcl::PointWithScale> bs_keypoints;
        pcl::PointCloud<pcl::PointWithScale> rw_keypoints;

        //get sift keypoints
        sift.setInputCloud(bs_cloud);
        sift.setSearchSurface(bs_cloud);
        sift.compute(bs_keypoints);
        sift.setInputCloud(rw_cloud);
        sift.compute(rw_keypoints);


        if (bs_keypoints.points.size() == 0) {
            outInfo("Bs Keypoints ist leer");
            //continue;
        }
        if (rw_keypoints.points.size() == 0) {
            outInfo("Rw Keypoints ist leer");
            //continue;
        }

        outInfo("bscloudkeypoints " << bs_keypoints.points.size());
        outInfo("rwcloudkeypoints " << rw_keypoints.points.size());

        if (save_images_for_debug) {
            pcl::io::savePCDFileBinary(
                    "/home/ros/Desktop/Aufnahmen/PCL/rw_" + std::to_string(iteration) + "_" + classification +
                    "_scene_keypointcloud.pcd", rw_keypoints);
            pcl::io::savePCDFileBinary(
                    "/home/ros/Desktop/Aufnahmen/PCL/bs_" + std::to_string(iteration) + "_" + classification +
                    "_scene_keypointcloud.pcd", bs_keypoints);
        }
        pcl::PointCloud<pcl::PointWithScale>::Ptr merged_keypoints_pcl_ptr(
                new pcl::PointCloud<pcl::PointWithScale>());;
        pcl::PointCloud<pcl::PointWithScale> merged_keypoints_pcl;


        //copy to ptr for fpfh
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr bs_keypoints_ptr(new pcl::PointCloud<pcl::PointXYZRGBA>);
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr rw_keypoints_ptr(new pcl::PointCloud<pcl::PointXYZRGBA>);
        copyPointCloud(bs_keypoints, *bs_keypoints_ptr);
        copyPointCloud(rw_keypoints, *rw_keypoints_ptr);
        bs_keypoint_map_3d[classification] = bs_keypoints_ptr;
        rw_keypoint_map_3d[classification] = rw_keypoints_ptr;

        // we need normal estimation here to get thenormals of the pointcloudsnippets and not the normals of the whole cloud.
        //normal esimation ala https://vml.sakura.ne.jp/koeda/PCL/tutorials/html/normal_estimation.html
        pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> ne;

        pcl::PointCloud<pcl::Normal>::Ptr bs_norm(new pcl::PointCloud<pcl::Normal>);
        pcl::PointCloud<pcl::Normal>::Ptr rw_norm(new pcl::PointCloud<pcl::Normal>);

        pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr bs_norm_tree(new pcl::search::KdTree<pcl::PointXYZRGBA>());
        pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr rw_norm_tree(new pcl::search::KdTree<pcl::PointXYZRGBA>());

//get normale for clouds
        ne.setInputCloud(bs_cloud);
        ne.setSearchMethod(bs_norm_tree);
        ne.setRadiusSearch(0.05);
        ne.compute(*bs_norm);

        ne.setInputCloud(rw_cloud);
        ne.setSearchMethod(rw_norm_tree);
        ne.setRadiusSearch(0.05);
        ne.compute(*rw_norm);




        //fpfh from here to map the keypoints with normale
        pcl::FPFHEstimation<pcl::PointXYZRGBA, pcl::Normal, pcl::FPFHSignature33> fpfh_estimation;
        pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBA>);
        fpfh_estimation.setSearchMethod(tree);
        fpfh_estimation.setRadiusSearch(0.06);

        pcl::PointCloud<pcl::FPFHSignature33>::Ptr bs_stuff(new pcl::PointCloud<pcl::FPFHSignature33>());
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr rw_stuff(new pcl::PointCloud<pcl::FPFHSignature33>());

        fpfh_estimation.setInputCloud(bs_keypoints_ptr);
        fpfh_estimation.setSearchSurface(bs_cloud->makeShared());
        fpfh_estimation.setInputNormals(bs_norm);
        fpfh_estimation.compute(*bs_stuff);

        fpfh_estimation.setInputCloud(rw_keypoints_ptr);
        fpfh_estimation.setSearchSurface(rw_cloud->makeShared());
        fpfh_estimation.setInputNormals(rw_norm);
        fpfh_estimation.compute(*rw_stuff);

//mapping here
        //https://docs.ros.org/hydro/api/pcl/html/classpcl_1_1registration_1_1CorrespondenceEstimation.html
        pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> est;
        est.setInputSource(bs_stuff);
        est.setInputTarget(rw_stuff);

        pcl::CorrespondencesPtr corr(new pcl::Correspondences);
        // Determine all reciprocal correspondences
        est.determineReciprocalCorrespondences(*corr);
        outInfo("CorrespondenceEstimation: Found " + std::to_string(corr->size()) + " Correspondences");

        for (auto co: *corr) {
            outInfo(co.distance);
        }

        //reject first corrs via distance
        auto rejectdist = pcl::registration::CorrespondenceRejectorDistance();
        rejectdist.setMaximumDistance(23);
        pcl::CorrespondencesPtr corrs_afterDist(new pcl::Correspondences);
        rejectdist.getRemainingCorrespondences(*corr, *corrs_afterDist);
        outInfo(" Dist: " + std::to_string(corrs_afterDist->size()) + " Correspondences");

        for (auto co: *corrs_afterDist) {
            outInfo(co.distance);
        }

        //validate /reject rest via ransac
        double inlier_threshold = 0.1;
        int iterations = 10000;
        pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGBA> sac;
        sac.setInputSource(bs_cloud);
        sac.setInputTarget(rw_cloud);
        sac.setInlierThreshold(inlier_threshold);
        sac.setMaximumIterations(iterations);
        sac.setRefineModel(true);
        //  sac.setInputCorrespondences(corrs_afterDist);
        pcl::Correspondences corrs;
        sac.getRemainingCorrespondences(*corrs_afterDist, corrs);
        outInfo(" RANSAC: " + std::to_string(corrs.size()) + " Correspondences");

        // double epsilon_sac = 0.001; // 10cm
        /*double inlier_threshold = 3.0;
        int iterations = 10000;
        pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGBA> sac;
        sac.setInputSource(bs_cloud);
        sac.setInputTarget(rw_cloud);
        sac.setInlierThreshold(inlier_threshold);
        sac.setMaximumIterations(iterations);
        sac.setRefineModel(true);
        sac.setInputCorrespondences(corr);


        boost::shared_ptr<pcl::Correspondences> cor_inliers_ptr(new pcl::Correspondences);
        sac.getCorrespondences(*cor_inliers_ptr);
        pcl::Correspondences corrs;
        corrs = *cor_inliers_ptr;
        outInfo(" RANSAC: " + std::to_string(cor_inliers_ptr->size()) + " Correspondences");*/

        /*auto rejectdist = pcl::registration::CorrespondenceRejectorDistance();
        rejectdist.setMaximumDistance(10);
        pcl::Correspondences corrs_afterDist;
        rejectdist.getRemainingCorrespondences(corrs, corrs_afterDist);
        outInfo(" Dist: " + std::to_string(corrs_afterDist.size()) + " Correspondences");*/


        corr_per_object[classification] = corrs;

//save remaining keypoint amount to cas
        rs::SceneCas cas(tcas);
        rs::Scene scene = cas.getScene();
        std::string classNameToSave = id_to_classname_[classification];
        if(mockClassification != "") {
            classNameToSave = mockClassification;
        }

        std::vector<image_similarity::SimilarityObject> imsimobjects;
        scene.annotations.filter(imsimobjects);
        bool contained = false;
        for (image_similarity::SimilarityObject iso: imsimobjects) {
            if (iso.classification.get() == classNameToSave) {
                iso.matches_pcl(corrs.size());
                contained = true;
                break;
            }
        }
        if (contained == false) {
            image_similarity::SimilarityObject imsimObject = rs::create<image_similarity::SimilarityObject>(
                    tcas);
            imsimObject.classification.set(classNameToSave);
            imsimObject.matches_pcl(corrs.size());
            scene.annotations.append(imsimObject);
        }

    }
    void colorHist_new(CAS &tcas, vector<rs::ColorHistogram> bs_hist_vec, vector<rs::ColorHistogram> rw_hist_vec , std::string classification, std::string mockClassification) {

        rs::ColorHistogram bs_hist = bs_hist_vec[0];
        rs::ColorHistogram rw_hist = rw_hist_vec[0];
        // use rs colorhist compare function
        double result = rs::compare(bs_hist, rw_hist);


        rs::SceneCas cas(tcas);
        rs::Scene scene = cas.getScene();

        std::string classNameToSave = id_to_classname_[classification];
        if(mockClassification != "") {
            classNameToSave = mockClassification;
        }

        //add colorhist result the cas if the object already exists
        std::vector<image_similarity::SimilarityObject> imsimobjects;
        scene.annotations.filter(imsimobjects);
        bool contained = false;
        for (image_similarity::SimilarityObject iso: imsimobjects) {
            if (iso.classification.get() == classNameToSave) {
                iso.colorHist_distance(result);
                contained = true;
                break;
            }
        } //if it does not exist create a new one
        if (contained == false) {
            image_similarity::SimilarityObject imsimObject = rs::create<image_similarity::SimilarityObject>(
                    tcas);
            imsimObject.classification.set(classNameToSave);
            imsimObject.colorHist_distance(result);
            scene.annotations.append(imsimObject);
        }
        outInfo("Colorhist distance " << result);
    }

    void colorhist(CAS &tcas, cv::Rect roi, cv::Mat mask, std::vector<rs_bs::BeliefStateObject> clusters, size_t idx,
                   rs::Scene scene, std::string mockClassification) {
        //Code written by Patrick Mania in https://github.com/RoboSherlock/robosherlock/blob/master/robosherlock/src/annotation/src/ClusterColorHistogramCalculator.cpp
        // and edited to work with newer changes by Tim Kratky


        uima::CAS *other_cas;
        other_cas = rs::CASConsumerContext::getInstance().getCAS(other_cas_id);
        if (!other_cas) {
            outError("Couldn't fetch CAS identified by '"
                             << other_cas_id
                             << "'. Make sure you have loaded an AAE with that name and "
                             << " that you've set 'otherCASId' in this config");
        }
        rs::SceneCas other_cas_(*other_cas);
        rs::Scene other_cas_scene = other_cas_.getScene();


        /*bool found = false;
        std::vector<rs_bs::BeliefStateObject> clusters;
        scene.identifiables.filter(clusters);

        cluster_rois_.resize(clusters.size());
        color_ids_.resize(clusters.size(), std::vector<int>(COUNT));
        color_ratios_.resize(clusters.size(), std::vector<float>(COUNT));
        for (size_t idx = 0; idx < clusters.size(); ++idx) {

        //    std::string classification = id_to_classname_[clusters[idx].rsObjectId.get()];
          //  rs::ImageROI image_rois = clusters[idx].rois.get();

            cv::Rect roi = bs_roi_map[clusters[idx].rsObjectId.get()];
            cv::Mat mask = bs_rois_with_mask[clusters[idx].rsObjectId.get()];*/

        //======================= Calculate HSV image ==========================
        cv::Mat rgb;

        outInfo("masksize " << mask.size);
        outInfo("roisize " << roi.size());
        outInfo("rgb_size " << rgb_.size());
        outInfo("rgbsize " << rgb.size());
       // outInfo("maskroisize " << mask(roi).size);
        outInfo("rgb_roisize " << rgb_(roi).size);
       /* int cols = mask.cols;
        int rows = mask.rows;
        outInfo(mask.type());
        //the whole calculation crashes if mask and roi do not work with each other or roi is out of bound for mask
        if(mask.cols == roi.height && mask.rows == roi.width) {
            rgb_.copyTo(rgb, mask);
        } else if(mask.cols == roi.width && mask.rows == roi.height){
            rgb_(roi).copyTo(rgb);
        }
        else {
            rgb_(roi).copyTo(rgb, mask(roi));
        }*/
       if(roi.width != mask.size().width || roi.height != mask.size().height) {
           outInfo("roi.width " <<roi.width);
           outInfo("roi.height " <<roi.height);
           outInfo("mask.size().width " <<mask.size().width);
           outInfo(" mask.size().height " << mask.size().height);
           mask.t();
           outInfo("roi.width " <<roi.width);
           outInfo("roi.height " <<roi.height);
           outInfo("mask.size().width " <<mask.size().width);
           outInfo(" mask.size().height " << mask.size().height);
       }
        assert(roi.width == mask.size().width);
        assert(roi.height == mask.size().height);
        rgb_(roi).copyTo(rgb, mask);




        cv::Mat hsv, hist;
        cv::cvtColor(rgb, hsv, CV_BGR2HSV_FULL);
        size_t sum;
        std::vector<int> colorCount;
        countColors(hsv, mask, colorCount, sum);

        //======================= Calculate Semantic Color ==========================
        if (semantic_label_) {
            std::vector<std::tuple<int, int>> colorsVec(COUNT);
            for (int i = 0; i < COUNT; ++i) {
                colorsVec[i] = std::tuple<int, int>(i, colorCount[i]);
            }

            std::sort(colorsVec.begin(), colorsVec.end(),
                      [](const std::tuple<int, int> &a, const std::tuple<int, int> &b) {
                          return std::get<1>(a) > std::get<1>(b);
                      });

            std::vector<int> &ids = color_ids_[idx];
            std::vector<float> &ratios = color_ratios_[idx];
            std::vector<std::string> colors(COUNT);


            for (size_t i = 0; i < COUNT; ++i) {
                int id, ratio_i;
                std::tie(id, ratio_i) = colorsVec[i];
                ids[i] = id;
                colors[i] = color_names_[id];

                std::string color = color_names_[id];

                float ratio_f = (float) (ratio_i / (double) sum);
                ratios[i] = ratio_f;

                if (ratio_f > 0.2) {
                    rs::SemanticColor colorAnnotation = rs::create<rs::SemanticColor>(tcas);
                    colorAnnotation.color.set(color);
                    colorAnnotation.ratio.set(ratio_f);
                    clusters[idx].annotations.append(colorAnnotation);
                }
            }
        }
        //======================= Calculate Color Histogram ==========================
        //Create the histogram
        int histSize[] = {histogram_cols_, histogram_rows_};
        float hranges[] = {0, 256}; // hue varies from 0 to 255, see cvtColor
        float sranges[] = {0, 256}; // saturation varies from 0 (black-gray-white) to 255 (pure spectrum color)
        const float *ranges[] = {hranges, sranges};

        // we compute the histogram from the 0-th and 1-st channels
        int channels[] = {0, 1};
        outInfo("hsv " << hsv.size << " " << hsv.depth() << " " << hsv.type());
        outInfo("mask " << mask.size << " " << mask.depth()<< " " << mask.type());
        outInfo("maskroi " << mask.size << " " << mask.depth()<< " " << mask.type());

        cv::calcHist(&hsv, 1, channels, mask, hist, 2, histSize, ranges, true, false);

        //Normalize histogram
        for (int r = 0; r < hist.rows; ++r) {

            float *it = hist.ptr<float>(r);
            for (int c = 0; c < hist.cols; ++c, ++it) {
                *it /= sum;
            }
        }

        rs::ColorHistogram color_hist_annotation = rs::create<rs::ColorHistogram>(tcas);
        color_hist_annotation.hist.set(rs::conversion::to(tcas, hist));
        outDebug("Containers for annotations created");
        compareWithRWColorHist(color_hist_annotation, scene, other_cas_scene, tcas, mockClassification);
        clusters[idx].annotations.append(color_hist_annotation);
        //}
    }

    //code written by Patrick Mania in https://github.com/RoboSherlock/robosherlock/blob/master/robosherlock/src/annotation/src/ClusterColorHistogramCalculator.cpp
    void countColors(const cv::Mat &hsv, const cv::Mat &mask, std::vector<int> &colorCount, size_t &sum) const {
        assert(hsv.type() == CV_8UC3);

        sum = 0;
        colorCount = std::vector<int>(COUNT, 0.0);

        for (int r = 0; r < hsv.rows; ++r) {
            const cv::Vec3b *itHSV = hsv.ptr<cv::Vec3b>(r);
            const uint8_t *itM = mask.ptr<uint8_t>(r);

            for (int c = 0; c < hsv.cols; ++c, ++itHSV, ++itM) {
                if (!*itM) {
                    continue;
                }

                ++sum;
                const uint8_t hue = itHSV->val[0];
                const uint8_t sat = itHSV->val[1];
                const uint8_t val = itHSV->val[2];

                if (sat > min_saturation_color_ && val > min_value_color_) {
                    if (hue < color_positions_[RED]) {
                        ++colorCount[RED];
                    } else if (hue < color_positions_[YELLOW]) {
                        ++colorCount[YELLOW];
                    } else if (hue < color_positions_[GREEN]) {
                        ++colorCount[GREEN];
                    } else if (hue < color_positions_[CYAN]) {
                        ++colorCount[CYAN];
                    } else if (hue < color_positions_[BLUE]) {
                        ++colorCount[BLUE];
                    } else if (hue < color_positions_[MAGENTA]) {
                        ++colorCount[MAGENTA];
                    } else {
                        ++colorCount[RED];
                    }
                } else if (val <= max_value_black_) {
                    ++colorCount[BLACK];
                } else if (val > min_value_white_) {
                    ++colorCount[WHITE];
                } else {
                    ++colorCount[GREY];
                }
            }
        }
        //End of Patrickscode
    }

    void compareWithRWColorHist(rs::ColorHistogram bs_ch, rs::Scene scene, rs::Scene other_cas_scene, CAS &tcas, std::string mockClassification) {

        std::vector<rs::ObjectHypothesis> clusters;
        other_cas_scene.identifiables.filter(clusters);
        outInfo(clusters.size());
        outInfo("colorhistdist");
        std::string class_name = "";
        for (rs::ObjectHypothesis oh : clusters) {
            std::vector<rs::ColorHistogram> colorHistograms;
            oh.annotations.filter(colorHistograms);


            std::vector<rs::Classification> classes;
            oh.annotations.filter(classes);
            class_name = classes[0].classname.get();

            for (rs::ColorHistogram rwHist : colorHistograms) {
                double dist = rs::compare(rwHist, bs_ch);
                outInfo(oh.id.get());
                outInfo(class_name);

                std::string classNameToSave = class_name;
                if(mockClassification != "") {
                    classNameToSave = mockClassification;
                }

                std::vector<image_similarity::SimilarityObject> imsimobjects;
                scene.annotations.filter(imsimobjects);
                bool contained = false;
                for (image_similarity::SimilarityObject iso: imsimobjects) {
                    if (iso.classification.get() == classNameToSave) {
                        iso.colorHist_distance(dist);
                        contained = true;
                        break;
                    }
                }
                if (contained == false) {
                    image_similarity::SimilarityObject imsimObject = rs::create<image_similarity::SimilarityObject>(
                            tcas);
                    imsimObject.classification.set(classNameToSave);
                    imsimObject.colorHist_distance(dist);
                    scene.annotations.append(imsimObject);
                }
            }
        }
    }

    //this method loads the trainingsdata and trains the randomforest
    void loadTrainingDataAndTrain() {

        cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::loadFromCSV(training_data_path, 1, 0, 1);
        if(useRTree) {
            rndTree = cv::ml::RTrees::create();
            rndTree->setMaxDepth(max_depth);
            rndTree->setMinSampleCount(min_samples_split);
            rndTree->setMaxCategories(max_features);
            rndTree->train(tdata);
        }
        else {
            logReg = cv::ml::LogisticRegression::create();
            logReg->setIterations(100);
            logReg->setRegularization(cv::ml::LogisticRegression::RegKinds::REG_L2);
          //  logReg->setTrainMethod(cv::ml::LogisticRegression::Methods::MINI_BATCH);
            //logReg->setMiniBatchSize(10);
            logReg->train(tdata);
        }
    }

    //thestmethod to get the accurecy create a confusionmatrix with the trained randomforest. uses a different dataset then the training
    void validateTraining() {
        std::ifstream trainDataStream;
        cv::Mat results;
        cv::Ptr<cv::ml::TrainData> pdata = cv::ml::TrainData::loadFromCSV(classify_data_path, 1, 0, 1);
        cv::Mat y_predict, y_predict_conf;
        std::string classifierName ="";
        if(useRTree) {
            classifierName = "RandomForestClassifier";
            rndTree->predict(pdata->getSamples(), y_predict);
        } else {
            classifierName = "LogisticRegression";
            //logReg->predict(pdata->getSamples(), y_predict, ml::StatModel::RAW_OUTPUT);
            logReg->predict(pdata->getSamples(), y_predict);
            logReg->predict(pdata->getSamples(), y_predict_conf, ml::StatModel::RAW_OUTPUT);
            //logReg->predict()
        }
        trainDataStream.open(ground_truth_path);

        int true_positive = 0;
        int true_negative = 0;
        int false_positive = 0;
        int false_negative = 0;
        std::string line;
        float val;
        // Keep track of the current column index
        int colIdx = 0;

        while (std::getline(trainDataStream, line)) {
            // Create a stringstream of the current line
            std::stringstream ss(line);

            // Extract each integer
            while (ss >> val) {
                // outInfo("val:" << val);
                //  outInfo("pred: " << std::to_string(y_predict.at<float>(colIdx, 0)));
                float pred = y_predict.at<float>(colIdx, 0);
                if(!useRTree) {
                    float conf = y_predict_conf.at<float>(colIdx, 0);
                    outInfo("collidx " << colIdx << " val " << val << " pred " << pred << " conf " << conf);
                } else {
                    outInfo("collidx " << colIdx << " val " << val << " pred " << pred);
                }
                /*if(useRTree) {
                    pred = y_predict.at<float>(colIdx, 0);
                } else {
                 pred = y_predict.at<float>(colIdx,0);
                }*/
              //  outInfo(y_predict.at<int>(0,0));
               // outInfo(y_predict.at<int>(0,1));
               /* if(!useRTree) {//to enable a confmatrx for logistic regression
                    //outInfo(pred);
                    if(pred >= logRegThreshold) {
                        pred = 1;
                    }else {
                        pred = 0;
                    }
                    /*if(pred<=0.25) {
                        pred =0;
                    }*/
                //}
                if (val == 1 and pred == 1) {
                    true_positive++;
                }
                if (val == 0 and pred == 0) {
                    true_negative++;
                }
                if (val == 0 and pred == 1) {
                    false_positive++;
                }
                if (val == 1 and pred == 0) {
                    false_negative++;
                }

            }
            colIdx++;
        }

        outInfo("Confusionmatrix for " << classifierName << " Training in Similarity Summarizer:");
        outInfo("    predneg    predpos     ");
        outInfo("neg    " << true_negative << "         " << false_positive);
        outInfo("pos    " << false_negative << "        " << true_positive);

        float right_ones = true_negative + true_positive;
        float false_ones = false_positive + false_negative;
        float all = right_ones + false_ones;
        float accuracy_tree = right_ones / all;
        outInfo("Accuracy:  " << accuracy_tree);
    }

    //mocks analyis engine from robosherlockf objects where not identical in first validation
    void pipelineMockForKNNNeighbors(std::vector<std::string> foundClasses, CAS &tcas, std::string comparedTo) {
        Ptr<SURF> surf_ptr = SURF::create(500);
        surf_ptr->setExtended(true);
        BFMatcher matcher(NORM_L1);
        std::string neighbors = "";

        std:
        vector<std::string> foundClassesWithoutFirst;
        //remove first entry of foundClasses (it is already spawned in the real bs pipeline and not needed to be mocked
        for (std::string className : foundClasses) {
            neighbors = neighbors + className + ";";
            if (className != comparedTo && classificationSpawnedInUE != className) {
                foundClassesWithoutFirst.push_back(className);
            }
                //special case. happens if for example comparedTo is albijuice but knn "thinks" that kellogs as a higher cofidence.
                //the kellogs would be spawned in ue, the pipeline would say it is not similar and without this albi would not be used to compare with.
            else if (classificationSpawnedInUE != comparedTo && className == comparedTo) {
                foundClassesWithoutFirst.push_back(className);
            }
        }
//only compare once with each classification
        vector<std::string> alreadyCompared;
        //for (std::string className : foundClasses) {
        for (std::string className : foundClassesWithoutFirst) {
            if (!(std::find(alreadyCompared.begin(), alreadyCompared.end(), className) != alreadyCompared.end())) {
                alreadyCompared.push_back(className);
            //if((className != comparedTo && classificationSpawnedInUE == comparedTo) || (classificationSpawnedInUE != comparedTo && className == comparedTo)) {
            if ((className != comparedTo) || (classificationSpawnedInUE != comparedTo && className == comparedTo)) {
                outInfo(className);
//loads image from disk to compare the realworld object with and mock rerun in ue with new object spawned
                std::vector<rs_bs::BeliefStateObject> bs_vec_for_colorHist = mockMoreBSData(tcas,
                                                                                            foundClassesWithoutFirst);
                for (std::string classification : list_of_object_classifications_surf) { //funktioniert nur da es eig nur ein objekt in der liste geben sollte
                  //  outInfo("vor filter " << id_to_classname_[classification]);
                  //  if (className == id_to_classname_[classification]) {
                    outInfo(classification);
                    outInfo(id_to_classname_[classification]);
                    cv::Rect rw_roi = rw_roi_map[classification];
                    if (rw_roi.empty()) {
                        outInfo("rw roi empty");
                        continue;
                    }
                    //cut interesting part out of the full size image
                    cv::Mat rw_cluster = other_cas_img_(rw_roi);

                    cv::Mat bs_cluster = imread(
                            "/home/ros/Desktop/Aufnahmen/Pipeline/JPG/bs_" + className + ".jpg");

                    if (bs_cluster.empty()) {
                        outInfo("No bs cluster read from file");
                        continue;
                    }
//run surf
                    surf(surf_ptr, matcher, tcas, bs_cluster, rw_cluster, className, false, className);

                    //run colorhist
                    rs::SceneCas cas(tcas);
                    rs::Scene scene = cas.getScene();
                    cluster_rois_.resize(bs_vec_for_colorHist.size());
                    color_ids_.resize(bs_vec_for_colorHist.size(), std::vector<int>(COUNT));
                    color_ratios_.resize(bs_vec_for_colorHist.size(), std::vector<float>(COUNT));
                    for (size_t idx = 0; idx < bs_vec_for_colorHist.size(); ++idx) {
                        if(bs_vec_for_colorHist[idx].rsObjectId.get() == className) {
                            //    std::string classification = id_to_classname_[clusters[idx].rsObjectId.get()];
                            //  rs::ImageROI image_rois = clusters[idx].rois.get();
                            outInfo(bs_vec_for_colorHist[idx].rsObjectId.get());
                            //cv::Rect roi2 = bs_roi_map[bs_vec_for_colorHist[idx].rsObjectId.get()];
                            //cv::Mat mask2 = bs_rois_with_mask[bs_vec_for_colorHist[idx].rsObjectId.get()];

                            cv::Rect roi;
                            cv::Mat mask;
                            rs::ImageROI image_rois = bs_vec_for_colorHist[idx].rois.get();
                            rs::conversion::from(image_rois.roi(), roi);
                            rs::conversion::from(image_rois.mask(), mask);

                            if (roi.empty()) {
                                outInfo("Roi empty");
                                continue;
                            }
                            if (mask.empty()) {
                                outInfo("Mask empty");
                                continue;
                            }
                            colorhist(tcas, roi, mask, bs_vec_for_colorHist, idx, scene, className);
                        }
                    }

                    //icp pcl
                    pcl::PointCloud<PointT>::Ptr rw_cloud = rw_pcl_map[classification];
                    pcl::PointCloud<PointT>::Ptr bs_cloud(new pcl::PointCloud<PointT>());
                    pcl::io::loadPCDFile<PointT>("/home/ros/Desktop/Aufnahmen/Pipeline/PCL/bs_" + className + ".pcd",
                                                 *bs_cloud);
                    if (bs_cloud->points.size() == 0) {
                        outInfo("No pointcloud loaded to bs_cloud from file");
                        continue;
                    }
                    if (rw_cloud->points.size() == 0) {
                        outInfo("No pointcloud loaded to rw_cloud");
                        continue;
                    }
                    pcl_pipeline(tcas, bs_cloud, rw_cloud, className, className);
                    icp_pipeline(tcas, bs_cloud, rw_cloud, className, className);
                    }
        //        }
            }
            }
        }
                rs::SceneCas cas(tcas);
                rs::Scene scene = cas.getScene();

                std::vector<image_similarity::SimilarityObject> imsimobjects;
                scene.annotations.filter(imsimobjects);
                if (imsimobjects.size() == 0) {
                    outInfo("No objects found.");
                }
//write results in csv
                for (image_similarity::SimilarityObject iso : imsimobjects) {
                     outInfo(iso.classification.get());
                    if (!(std::find(alreadyWrittenToCsv.begin(), alreadyWrittenToCsv.end(), iso.classification.get()) != alreadyWrittenToCsv.end())) {
                    //    alreadyWrittenToCsv.push_back(iso.classification.get());
                     if((iso.classification.get() != comparedTo ) || (classificationSpawnedInUE != comparedTo && iso.classification.get() == comparedTo)) {
                         //if(iso.classification.get() != comparedTo) {
                         bool result;
                         float confidence;
                         if (useRTree) { //use randomforest to check if this time the objects where identical
                             result = randomForest(iso);
                             confidence = -1; //confidence -1 => not set / used
                         } else {
                             const std::pair<bool, float> &resultPair = logisticRegression(iso);
                             result = resultPair.first;
                             confidence = resultPair.second;
                         }
                         bool imres = false;
                         if (iso.classification.get() == comparedTo) {
                             imres = true;
                         }
                         writeToCsv(iso, comparedTo, imres, result, confidence, true, neighbors);
                         if(result) { //Interrupting the writing loop because we are only interested in the first positiv guess
                             break;
                         }
                     }
                    }
                }

    }
// returns found classifications from cas with confidenz
    std::pair<std::vector<std::string>, std::string> getKNNNeighborClasses(CAS &tcas) {
        rs::SceneCas cas(tcas);
        rs::Scene scene = cas.getScene();

        uima::CAS *other_cas;
        other_cas = rs::CASConsumerContext::getInstance().getCAS(other_cas_id);
        if (!other_cas) {
            outError("Couldn't fetch CAS identified by '"
                             << other_cas_id
                             << "'. Make sure you have loaded an AAE with that name and "
                             << " that you've set 'otherCASId' in this config");
        }
        rs::SceneCas other_cas_(*other_cas);
        rs::Scene other_cas_scene = other_cas_.getScene();

        std::string neighborsAndDistances = "";
        std::vector<std::string> foundClasses;

        //get knnneighbors via othercas
        std::vector<rs::ObjectHypothesis> objectHypothesises;
        other_cas_scene.identifiables.filter(objectHypothesises);
        if (objectHypothesises.size() == 0) {
            outInfo("No ObjectHypothesis found.");
        }
        for (rs::ObjectHypothesis oh : objectHypothesises) {
            std::vector<rs::Classification> classifiedNeighbors;
            outInfo(oh.id.get());
            oh.annotations.filter(classifiedNeighbors);
            if (classifiedNeighbors.size() == 0) {
                outInfo("No Classification found.");
            }
            for (rs::Classification cn : classifiedNeighbors) {
                for (auto conf  : cn.confidences.get()) {
                    outInfo(conf.name.get());
                    outInfo(conf.score.get());
                    if (std::find(foundClasses.begin(), foundClasses.end(), conf.name.get()) == foundClasses.end()) {
                        //we are only interested in one occurence of a classification.
                        // so if knn returns one classification more then once it will still only be used one time here
                        foundClasses.push_back(conf.name.get());
                        neighborsAndDistances =
                                neighborsAndDistances + conf.name.get() + " " + std::to_string(conf.score.get()) + ";";
                    }
                }
            }
        }
        return std::make_pair(foundClasses, neighborsAndDistances);
    }

    //run randomforest classifier on single datset and predict if the images where similar or not.
    bool randomForest(image_similarity::SimilarityObject iso) {
        float data[4] = {float(iso.matches_surf.get()), float(iso.matches_pcl.get()),
                         iso.similarityscore_icp.get(), iso.colorHist_distance.get()};
       // float data[3] = {float(iso.matches_surf.get()), float(iso.matches_pcl.get()), iso.colorHist_distance.get()};
        //cv::Mat dataToPredictOn = cv::Mat(1, 3, CV_32F, data);
        cv::Mat dataToPredictOn = cv::Mat(1, 4, CV_32F, data);
        cv::Mat prediction;
        rndTree->predict(dataToPredictOn, prediction);
        outInfo("prediction: " << prediction.at<float>(0,0));

      int resultAsInt = prediction.at<float>(0, 0); //float is either 0 or 1. everything else should not occure here

        if (resultAsInt == 1) {
            return true;
        } else {
            return false;
        }
    }

    //run logistic regression to predict if objects where identical
    std::pair<bool,float> logisticRegression(image_similarity::SimilarityObject iso) {
        float data[4] = {float(iso.matches_surf.get()), float(iso.matches_pcl.get()),
                         iso.similarityscore_icp.get(), iso.colorHist_distance.get()};
        cv::Mat dataToPredictOn = cv::Mat(1, 4, CV_32F, data);
        cv::Mat prediction,prediction_conf;
        logReg->predict(dataToPredictOn, prediction);
        logReg->predict(dataToPredictOn, prediction_conf, ml::StatModel::RAW_OUTPUT);
        int pred =  prediction.at<int>(0,0);
        float conf = prediction_conf.at<float>(0,0);
        outInfo("prediction: " << pred);
        outInfo("confidence: " << conf);

        return std::make_pair(pred, conf);
    }

//write to csv and print results to console for debug reasons
    void writeToCsv(image_similarity::SimilarityObject iso, std::string comparedTo, bool imres, bool result, float confidence, bool mock,
                    std::string neighborsAndDistances) {
        alreadyWrittenToCsv.push_back(iso.classification.get());
        myFile.open("/home/ros/Desktop/similarity.csv", std::ofstream::out | std::ofstream::app);

        outInfo("classification ");
        outInfo(iso.classification.get());
        outInfo("Compared_to");
        outInfo(comparedTo);
        outInfo("Surfmatches ");
        outInfo(iso.matches_surf.get());
        outInfo("PClmatches ");
        outInfo(iso.matches_pcl.get());
        outInfo("ICP ");
        outInfo(iso.similarityscore_icp.get());
        outInfo("ColorHistDist ");
        outInfo(iso.colorHist_distance.get());
        outInfo("guess ");
        outInfo(result);
        outInfo(" confidence if set (then sth differenct then -1) ");
        outInfo(confidence);
        outInfo("mockedPipeline");
        outInfo(mock);
        outInfo("neighborsWithDist ");
        outInfo(neighborsAndDistances);
        myFile << iso.classification.get() << "," << comparedTo << "," << imres << ","
               << iso.matches_surf.get()
               << "," << iso.matches_pcl.get() << "," << iso.similarityscore_icp.get() << ","
               << iso.colorHist_distance.get() << "," << result << "," << confidence << "," << mock << ","
               << neighborsAndDistances
               << std::endl;

        myFile.close();
    }

    //load images from storage to compare against them
    std::vector<rs_bs::BeliefStateObject> mockMoreBSData(CAS &tcas,std::vector<std::string> foundClasses) {
        std::vector<rs_bs::BeliefStateObject> all_beliefstate_objects;
        for(std::string className : foundClasses) {
        rs_bs::BeliefStateObject bsObject = rs::create<rs_bs::BeliefStateObject>(tcas);
        rs::ImageROI imageROI = rs::create<rs::ImageROI>(tcas);
        cv::Mat bs_mask;
        outInfo("mask before " << bs_mask.size << " " << bs_mask.depth()<< " " << bs_mask.type());
        bs_mask =imread("/home/ros/Desktop/Aufnahmen/Pipeline/JPG/mask_" + className + ".jpg", IMREAD_UNCHANGED);
        outInfo("mask read " << bs_mask.size << " " << bs_mask.depth()<< " " << bs_mask.type());
        bs_mask.convertTo(bs_mask, 	CV_8UC2);
        outInfo("mask after conv " << bs_mask.size << " " << bs_mask.depth()<< " " << bs_mask.type());
        masktwo =bs_mask;
        if(bs_mask.empty()) {
            outInfo("bs_mask empty is the file there?");
        }
        cv::Rect roi;
            FileStorage fs("/home/ros/Desktop/Aufnahmen/Pipeline/JPG/roi_" + className + ".yml", FileStorage::READ);
            if( fs.isOpened() ){
                fs["x"] >> roi.x;
                fs["y"] >> roi.y;
                fs["width"] >> roi.width;
                fs["height"] >> roi.height;
            }
        if(roi.empty()) {
            outInfo("roi empty is the file there?");
        }
        imageROI.mask(rs::conversion::to(tcas, bs_mask));
        imageROI.mask_hires(rs::conversion::to(tcas, bs_mask));
        imageROI.roi_hires(rs::conversion::to(tcas, roi));
        imageROI.roi(rs::conversion::to(tcas, roi));
        bsObject.rois.set(imageROI);

        bsObject.source.set("ImageSimilarityMock");

        // Write already available information to the annotation
       // bsObject.mask_color_b.set(blue);
        //bsObject.mask_color_g.set(green);
        //bsObject.mask_color_r.set(red);
        bsObject.simulationActorId.set(className);
        bsObject.rsObjectId.set(className);
        outInfo("set simulationActorId as " << className);
      //  bsObject.points.set(rcp);

        all_beliefstate_objects.push_back(bsObject);
    }
    return all_beliefstate_objects;
   // cas.set(VIEW_ALL_BS_OBJECTS, all_beliefstate_objects);
    }

    //method which runs random forest at the annotated data safed inside the similarityobject.
    //if random forest returns that the beliefstateobject was not similar with the realworld object it uses photos / saved pointclouds of the next knn-neighbor
    //to compare the realworld object with them until one seems to be the correct match.
    void runClassifierOnLiveDataAndWriteCsv(CAS &tcas) {
        rs::SceneCas cas(tcas);
        rs::Scene scene = cas.getScene();

        uima::CAS *other_cas;
        other_cas = rs::CASConsumerContext::getInstance().getCAS(other_cas_id);
        if (!other_cas) {
            outError("Couldn't fetch CAS identified by '"
                             << other_cas_id
                             << "'. Make sure you have loaded an AAE with that name and "
                             << " that you've set 'otherCASId' in this config");
        }
        rs::SceneCas other_cas_(*other_cas);
        rs::Scene other_cas_scene = other_cas_.getScene();

//gets classification from cas (either baselineclassifier or rsknn as source)
        auto pair = getKNNNeighborClasses(tcas);
        std::vector<std::string> foundClasses = pair.first;
        std::string neighborsAndDistances = pair.second;


        std::vector<image_similarity::SimilarityObject> imsimobjects;
        scene.annotations.filter(imsimobjects);
        if (imsimobjects.size() == 0) {
            outInfo("No objects found.");
        }



//atm nur ein objekt drin und für diese pipeline auch das einzig mögliche (sonst würde der quatsch mit den runden nicht gehen
        for (image_similarity::SimilarityObject iso : imsimobjects) {

            //"AlbiHimbeerJuice";
//"KelloggsCornFlakes";
//KoellnMuesliKnusperHonigNuss
//"VollMilch";
//"SpitzenReis";
//"PfannerGruneIcetea";
//"PfannerPfirsichIcetea"
//"SojaMilch";
            std::string comparedTo = "PfannerPfirsichIcetea";

            bool imres = false;

            if (comparedTo == iso.classification.get()) {
                imres = true;
            }

            int resultCounter = 0;
            bool result = false;
            outInfo("surf" << iso.matches_surf.get() << " pcl " << iso.matches_pcl.get() << " icp "
                                           << iso.similarityscore_icp.get()
                                           << " colorhist " << iso.colorHist_distance.get());

            float confidence;
            if(useRTree) { //use random forest for decision if objects where identical
                result = randomForest(iso);
                confidence = -1; //confidence -1 => not set / used
            } else { //use logistic regression for this descision (currently not working correctly)
                const std::pair<bool, float> &resultPair = logisticRegression(iso);
                result = resultPair.first;
                confidence = resultPair.second;

            }
            //write information to csv
            writeToCsv(iso, comparedTo, imres, result,confidence, false, neighborsAndDistances);

//if result is objects are not identical mock robosherlock pipeline and compare with other possible objects
         if ((!result && useRTree) || ( !useRTree && !result && confidence >= logRegThreshold) ) {
               pipelineMockForKNNNeighbors(foundClasses, tcas, comparedTo);
            }
          /*  else if( !useRTree && !result && confidence >= logRegThreshold) {
                pipelineMockForKNNNeighbors(foundClasses, tcas, comparedTo); //for logisitc regression usage
            }*/
        }
    }

//visualizise the pointclouds
    void visualizeKeyPointsAndCorr(pcl::visualization::PCLVisualizer &visualizer, std::string classification,
                                   bool withBackground) {
        if (nextMatch && (!(std::find(seenClassifications.begin(), seenClassifications.end(),
                                 classification) != seenClassifications.end()) ||
                     currentClassification == "")) {
            currentClassification = classification;
            seenClassifications.push_back(classification);
            nextMatch = false;
        }


        if (seenClassifications.size() == list_of_object_classifications_pcl.size() - 1) {
            seenClassifications.clear();
        }


        if (currentClassification != "") {
            auto bs = bs_keypoint_map_3d[currentClassification];
            auto rw = rw_keypoint_map_3d[currentClassification];

            if(bs->empty()) {
                outInfo("bs cloud empty");
            }
            if(rw == NULL) {
                outInfo("rw cloud empty");
            }

            pcl::visualization::PointCloudColorHandlerCustom<PointT> bs_color(bs, 255, 0, 0);
            visualizer.addPointCloud(bs, bs_color, "bsPoints");
            pcl::visualization::PointCloudColorHandlerCustom<PointT> rw_color(rw, 0, 255, 0);
            visualizer.addPointCloud(rw, rw_color, "rwPoints");
            visualizer.addCorrespondences<PointT>(bs, rw, corr_per_object[currentClassification]);
            if (withBackground) {
                visualizer.addPointCloud(bs_pcl_map[currentClassification], "bsCloud");
                visualizer.addPointCloud(rw_pcl_map[currentClassification], "rwCloud");

            }
        }
    }
//draw images as in alot of other annotators
    void drawImageWithLock(cv::Mat &disp) {
        if (!rgb_.empty()) {
            if (!other_cas_img_.empty()) {
                if(firstRun) {
                    firstRun = false;
                    dispMode = ALL_TOGETHER;
                }
                int classification_amount = 0;
                for (std::string classification : list_of_object_classifications_surf) {
                    if (classification != "unknown") {
                        switch (dispMode) {
                            case ALL_TOGETHER:
                                if (img_matches.empty() ||
                                    classification_amount < list_of_object_classifications_surf.size()) {
                                    cv::drawMatches(rgb_, keypoints_bs_compl[classification], other_cas_img_,
                                                    keypoints_rw_compl[classification],
                                                    complete_image_matches[classification],
                                                    img_matches,
                                                    Scalar::all(-1),
                                                    Scalar::all(-1), std::vector<char>(),
                                                    DrawMatchesFlags::DEFAULT);
                                    disp = img_matches;
                                    classification_amount = list_of_object_classifications_surf.size();
                                } else {
                                    cv::drawMatches(rgb_, keypoints_bs_compl[classification], other_cas_img_,
                                                    keypoints_rw_compl[classification],
                                                    complete_image_matches[classification],
                                                    img_matches,
                                                    Scalar::all(-1),
                                                    Scalar::all(-1), std::vector<char>(),
                                                    DrawMatchesFlags::DRAW_OVER_OUTIMG);
                                    disp = img_matches;
                                }
                                break;
                            case SINGLE_MATCHES:

                                if (nextMatch && (!(std::find(seenClassifications.begin(), seenClassifications.end(),
                                                              classification) != seenClassifications.end()) ||
                                                  currentClassification == "")) {
                                    currentClassification = classification;
                                    seenClassifications.push_back(classification);
                                    nextMatch = false;
                                }

                                if (seenClassifications.size() == list_of_object_classifications_surf.size() - 1) {
                                    seenClassifications.clear();
                                }

                                if (currentClassification != "") {
                                    cv::drawMatches(rgb_, keypoints_bs_compl[currentClassification], other_cas_img_,
                                                    keypoints_rw_compl[currentClassification],
                                                    complete_image_matches[currentClassification],
                                                    img_matches,
                                                    Scalar::all(-1),
                                                    Scalar::all(-1), std::vector<char>(),
                                                    DrawMatchesFlags::DEFAULT);
                                    disp = img_matches;
                                }
                                break;
                         /*   case COLOR_HIST:

                                disp = rgb_.clone();
                                if(true) { // ... ist halt schrott aber dann funktioniert es mit den cases dahinter
                                    //code written by Patrick Mania in https://github.com/RoboSherlock/robosherlock/blob/master/robosherlock/src/annotation/src/ClusterColorHistogramCalculator.cpp
                                    //for (size_t i = 0; i < cluster_rois_.size(); ++i) {
                                    //const cv::Rect &roi = cluster_rois_[i];
                                    const cv::Rect roi = bs_roi_map[classification];
                                    const cv::Size histSize(roi.width, 10);
                                    const cv::Rect roiHist(roi.x, roi.y + roi.height + 1, histSize.width,
                                                           histSize.height);
                                    const std::vector<int> &ids = color_ids_[0];
                                    const std::vector<float> &ratios = color_ratios_[0];

                                    cv::rectangle(disp, roi, colors_[ids[0]]);

                                    cv::Mat hist = disp(roiHist);

                                    float start = 0;
                                    for (int r = 0; r < ratios.size(); ++r) {
                                        float width = (histSize.width * ratios[r]);
                                        const cv::Rect rect(start + 0.5, 0, width + 0.5, histSize.height);
                                        start += width;
                                        cv::rectangle(hist, rect, colors_[ids[r]], CV_FILLED);
                                    }
                                    //End of Patrickscode
                                }
                               // }
                                break;*/
                            case ALL_TOGETHER_CLOUD:
                                break;
                            case BS_CLOUD_ONLY:
                                break;
                            case OTHER_CAS_CLOUD_ONLY:
                                break;
                            case KEYPOINTS_CORRESPONDENCES_SINGLE_OBJECT:
                                break;
                            case KEYPOINTS_CORRESPONDENCES_SINGLE_OBJECT_WITH_OBJECT:
                                break;
                            case MASKONE:
                                disp = maskone.clone();
                                break;
                            case MASKTWO:
                                disp = masktwo.clone();
                                break;
                        }
                    }
                }
            } else {
                outWarn("Other CAS img is empty.");
                return;
            }
        } else {
            outWarn("This CAS img is empty.");
            return;
        }
    }

    void fillVisualizerWithLock(pcl::visualization::PCLVisualizer &visualizer, const bool firstRun) {
        const std::string &cloudname = this->name;
        double pointsize = 5.0;
        if (!firstRun) {
            visualizer.getPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, pointsize,
                                                        cloudname);

        } else{
            visualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, pointsize, cloudname);
        }
        visualizer.removeAllPointClouds();
        visualizer.removeAllShapes();

        for (std::string classification : list_of_object_classifications_pcl) {
            if (classification != "unknown") {
                switch (dispMode) {
                    case ALL_TOGETHER_CLOUD:
                        visualizer.addPointCloud(dispCloud, cloudname);
                        break;
                    case BS_CLOUD_ONLY:
                        visualizer.addPointCloud(bs_cloud_, cloudname);
                        break;
                    case OTHER_CAS_CLOUD_ONLY:
                        visualizer.addPointCloud(other_cas_cloud_, cloudname);
                        break;
                    case KEYPOINTS_CORRESPONDENCES_SINGLE_OBJECT:
                        visualizeKeyPointsAndCorr(visualizer, classification, false);
                        break;
                    case KEYPOINTS_CORRESPONDENCES_SINGLE_OBJECT_WITH_OBJECT:
                        visualizeKeyPointsAndCorr(visualizer, classification, true);
                        break;
                    case ALL_TOGETHER:
                        break;
                    case SINGLE_MATCHES:
                        break;
                    case COLOR_HIST:
                        break;

                }
            }
        }
    }

//switch through the views
    bool callbackKey(const int key, const Source source) {
        switch (key) {
            case '1':
                dispMode = ALL_TOGETHER;
                break;
            case '2':
                dispMode = SINGLE_MATCHES;
                break;
            case '3':
                dispMode = SINGLE_MATCHES;
                nextMatch = true;
                break;
            case '4':
                dispMode = COLOR_HIST;
                break;
            case 'q':
                dispMode = ALL_TOGETHER_CLOUD;
                break;
            case 'w':
                dispMode = BS_CLOUD_ONLY;
                break;
            case 'e':
                dispMode = OTHER_CAS_CLOUD_ONLY;
                break;
            case 'r':
                dispMode = KEYPOINTS_CORRESPONDENCES_SINGLE_OBJECT;
                nextMatch = true;
                break;
            case 't':
                dispMode = KEYPOINTS_CORRESPONDENCES_SINGLE_OBJECT_WITH_OBJECT;
                nextMatch = true;
                break;
            case 'g':
                dispMode = MASKONE;
                break;
            case 'f':
                dispMode = MASKTWO;
                break;
        }
        return true;
    }

};

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(imsim)
