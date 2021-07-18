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

using namespace uima;
using namespace cv;
using namespace cv::xfeatures2d;


using std::cout;
using std::vector;


class ImageSimilarity : public DrawingAnnotator {
private:
    typedef pcl::PointXYZRGBA PointT;

    cv::Mat color, object_, rgb_, depth_, other_cas_img_, other_cas_depth_, img_matches, img_matches_test;
    pcl::PointCloud<PointT>::Ptr bs_cloud_;
    pcl::PointCloud<PointT>::Ptr other_cas_cloud_;
    vector<cv::KeyPoint> keypoints0, keypoints1;

    std::map<std::string, vector<cv::KeyPoint>> keypoints_bs_compl, keypoints_rw_compl;
    std::map<std::string, vector<cv::DMatch>> complete_image_matches;
    Mat descriptors0, descriptors1;
    vector<DMatch> matches;
    vector<DMatch> good_matches;
//    vector<DMatch> complete_image_matches;
//  std::vector<std::vector<DMatch> > knn_matches; //temp
    std::set<std::string> list_of_object_classifications;
    std::map<std::string, cv::Rect> bs_roi_map, rw_roi_map;
    std::map<std::string, vector<cv::KeyPoint>> bs_keypoint_map, rw_keypoint_map;
    //int classification_amount = 0;
    std::map<std::string, std::string> id_to_classname_;
    std::list<std::string> seenClassifications;
    std::string currentClassification = "";
    int iteration = -1;

    bool use_hd_images_ = true;
    bool save_images = true;
    bool nextMatch = false;
    enum {
        ALL_TOGETHER,
        SINGLE_MATCHES
    } dispMode;

    //
    // PARAMETERS
    //
    //
    // This identifier is used to reference the CAS of another AAE that ran before
    // *this* AAE. It is access via rs::CASConsumerContext
    std::string other_cas_id;

public:
    ImageSimilarity() : DrawingAnnotator(__func__) {

    }

    TyErrorId initialize(AnnotatorContext &ctx) {
        outInfo("initialize");
        cout << "OpenCV Version used:" << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << endl;
        if (ctx.isParameterDefined("otherCASId")) {
            ctx.extractValue("otherCASId", other_cas_id);
            outInfo("Using AAE/CAS identified by '" << other_cas_id << "' for imagesimilarity");
        }
        return UIMA_ERR_NONE;
    }

    TyErrorId destroy() {
        outInfo("destroy");
        return UIMA_ERR_NONE;
    }

    //searches the keypoints in the smaller image snippets which are later used to calculate the image similarity in the complete image. with this way there are
    // only keypoints detected at the objects we are interested in. if save_images_for_debug is true it will save the snippet images and an image with the matches
    void surfWithDebugImages(CAS &tcas) {
        //setup for image similarity via surf
        //Ptr<FastFeatureDetector> detector1= FastFeatureDetector::create(25, true);//45
        Ptr<SURF> surf = SURF::create(500);
        //Ptr<SURF> extractor = SURF::create();
        //extractor->setHessianThreshold(500); // vorher 500 (bester wert)
        //extractor->setExtended(true);
        surf->setExtended(true);
        BFMatcher matcher(NORM_L1);

        iteration++;
        for (std::string classification : list_of_object_classifications) {
            //if (classification != "unknown") { //we are only interested in classified stuff

            cv::Rect bs_roi = bs_roi_map[classification];
            cv::Rect rw_roi = rw_roi_map[classification];

            if (bs_roi.empty() || rw_roi.empty()) {
                outInfo("bs or rw roi empty");
                continue;
            }

            //cut interesting part out of the full size image
            cv::Mat bs_cluster = rgb_(bs_roi);
            cv::Mat rw_cluster = other_cas_img_(rw_roi);

            if (save_images) { //set save_images_for_debug for debug reasons or if you are interested in the snippet images and matching between snippet images
                imwrite("/home/ros/Desktop/Aufnahmen/JPG/bs_" + std::to_string(iteration) + "_" + classification +
                        "_cluster.jpg", bs_cluster);
                imwrite("/home/ros/Desktop/Aufnahmen/JPG/rw_" + std::to_string(iteration) + "_" + classification +
                        "_cluster.jpg", rw_cluster);
            }

            //detect keypoints
            surf->detect(bs_cluster, keypoints0);
            surf->detect(rw_cluster, keypoints1);

            if (keypoints0.size() == 0) {
                outInfo("bs keypoints size 0");
            }

            if (keypoints1.size() == 0) {
                outInfo("rw keypoints size 0");
            }

            //for complete image drawing
            bs_keypoint_map[classification] = keypoints0;
            rw_keypoint_map[classification] = keypoints1;


            if (save_images) { // this code here is only needed to draw the matches on the imagesnippets corresponding to each classification.
                // if the result is not saved there is not point in calculating everything for it

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
            }
        }
        //}

        //for the drawing inside the complete image and not only roi
        surf(surf, matcher, tcas);
    }

    void surf(Ptr<SURF> surf, BFMatcher matcher, CAS &tcas) {
        complete_image_matches.clear();
        keypoints_bs_compl.clear();
        keypoints_rw_compl.clear();


        for (std::string classification : list_of_object_classifications) {
            if (classification != "unknown") {
                vector<KeyPoint> bs_kp, rw_kp;
                bs_kp = bs_keypoint_map[classification]; //get image snippets for each classification
                rw_kp = rw_keypoint_map[classification];

                if (bs_kp.empty() || rw_kp.empty()) {
                    outInfo("bs or rw keypoints is empty");
                    continue;
                }

                cv::Rect bs_roi = bs_roi_map[classification]; // get rois for each classification
                cv::Rect rw_roi = rw_roi_map[classification];

                //add the roi coordinates to the keypoint coordinates. this is needed because we calculated the keypoints on the smaller image snippets to remove
                // unwanted background stuff and get better keypoints. now we put these found keypoints back into the original images to create a nice image with matches and keypoints
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

                //extract descriptors for matching
                surf->compute(rgb_, keypoints_bs_after_homography, descriptors0);
                surf->compute(other_cas_img_, keypoints_rw_after_homography, descriptors1);


                //matching descriptors
                matcher.match(descriptors0, descriptors1, matches);

                complete_image_matches[classification] = matches; // save matches for the view later


                rs::SceneCas cas(tcas);
                rs::Scene scene = cas.getScene();


                std::vector<image_similarity::SimilarityObject> imsimobjects;
                scene.annotations.filter(imsimobjects);
                bool contained = false;
                for (image_similarity::SimilarityObject iso: imsimobjects) {
                    if (iso.classification.get() == id_to_classname_[classification]) {
                        iso.matches_surf(matches.size());
                        contained = true;
                        break;
                    }
                }
                if (contained == false) {
                    image_similarity::SimilarityObject imsimObject = rs::create<image_similarity::SimilarityObject>(
                            tcas);
                    imsimObject.classification.set(id_to_classname_[classification]);
                    imsimObject.matches_surf(matches.size());
                    scene.annotations.append(imsimObject);
                }


                outInfo("amount of matches in complete image " << complete_image_matches.size());
            }
        }
    }


    TyErrorId processWithLock(CAS &tcas, ResultSpecification const &res_spec) {

        id_to_classname_.clear();

        outInfo("process start");
        rs::StopWatch clock;
        rs::SceneCas cas(tcas);

        bs_cloud_ = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
        other_cas_cloud_ = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
        // set everything for later
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
            return UIMA_ERR_ANNOTATOR_MISSING_INFO;
        }

        rs::SceneCas other_cas_scene(*other_cas);
        if (use_hd_images_) {
            other_cas_scene.get(VIEW_COLOR_IMAGE_HD, other_cas_img_);
            other_cas_scene.get(VIEW_DEPTH_IMAGE_HD, other_cas_depth_);
        } else {
            other_cas_scene.get(VIEW_COLOR_IMAGE, other_cas_img_);
            other_cas_scene.get(VIEW_DEPTH_IMAGE, other_cas_depth_);
        }
        other_cas_scene.get(VIEW_CLOUD, *other_cas_cloud_);

        if (rgb_.cols != other_cas_img_.cols || rgb_.rows != other_cas_img_.rows) {
            outError("BS and RW RGB image resolution Mismatch! "
                     "Resolutions of the other cas cam (" << other_cas_img_.cols << "x" << other_cas_img_.rows <<
                                                          ") and the synthetic views (" << rgb_.cols << "x" << rgb_.rows
                                                          << ") differ");
        }

        //from here rw stuff
        std::vector<rs::Object> rwobjects;
        other_cas_scene.get(VIEW_OBJECTS, rwobjects);
        outInfo("Found " << rwobjects.size() << " rs::Object in RW CAS");
        std::string mop = "";
        for (auto rwo: rwobjects) {

            cv::Rect rw_roi;
            std::string class_name;
            if (use_hd_images_) {
                rs::conversion::from(rwo.rois().roi_hires(), rw_roi);
            } else {
                rs::conversion::from(rwo.rois().roi(), rw_roi);
            }
            std::vector<rs::Classification> classes;
            rwo.annotations.filter(classes);
            outInfo("Found " << classes.size() << " object annotations in rwo");

            if (classes.size() == 0) {
                outInfo("No classification information for cluster ");
                class_name = "unknown";
            } else {
                class_name = classes[0].classname.get();
            }
            mop = rwo.id.get();
            id_to_classname_[rwo.id.get()] = class_name;
            list_of_object_classifications.insert(
                    rwo.id.get()); //fill this list for later. we will iterate often over it to get the image snippets per cluster and compare them
            rw_roi_map[rwo.id.get()] = rw_roi;


        }


        //from here bso stuff
        std::vector<rs_bs::BeliefStateObject> bsobjects;
        cas.get(VIEW_ALL_BS_OBJECTS, bsobjects);
        //scene.identifiables.filter(bsobjects);
        outInfo("Found " << bsobjects.size() << " rs::BeliefStateObject in BS CAS");

        for (auto bso: bsobjects) {

            cv::Rect bs_roi;

            if (use_hd_images_) {
                rs::conversion::from(bso.rois().roi_hires(), bs_roi);
            } else {
                rs::conversion::from(bso.rois().roi(), bs_roi);
            }

            // Fetch the class name from the matching RW cluster
            outInfo(bso.simulationActorId.get());
            outInfo(bso.rsObjectId.get());
            outInfo(bso.id.get());
            outInfo(mop);
            outInfo(bso.rsObjectId.get());
            if (bso.rsObjectId.get() == mop) {
                outInfo("hier");
            }
            outInfo("roi area  " << bs_roi.x << " " << bs_roi.y << " " << bs_roi.height << " " << bs_roi.width << "  simulationactorid  " << bso.simulationActorId.get());

            if (std::find(list_of_object_classifications.begin(), list_of_object_classifications.end(),
                          bso.rsObjectId.get()) != list_of_object_classifications.end()) {
                std::string class_name = id_to_classname_[bso.rsObjectId.get()];
                bs_roi_map[bso.rsObjectId.get()] = bs_roi;
            }

            //std::string class_name = id_to_classname_[bso.rsObjectId.get()];

            //list_of_object_classifications.insert(bso.rsObjectId.get());
            //bs_roi_map[bso.rsObjectId.get()] = bs_roi;
        }
        if (!bs_roi_map.empty()) {
            surfWithDebugImages(tcas);
        } else {
            outInfo("bs roi map empty");
        }
        outInfo("took: " << clock.getTime() << " ms.");
        return UIMA_ERR_NONE;

    }


    void drawImageWithLock(cv::Mat &disp) {
        if (!rgb_.empty()) {
            if (!other_cas_img_.empty()) {

                int classification_amount = 0;


                for (std::string classification : list_of_object_classifications) {
                    if (classification != "unknown") {


                        switch (dispMode) {
                            case ALL_TOGETHER:
                                if (img_matches.empty() ||
                                    classification_amount < list_of_object_classifications.size()) {
                                    cv::drawMatches(rgb_, keypoints_bs_compl[classification], other_cas_img_,
                                                    keypoints_rw_compl[classification],
                                                    complete_image_matches[classification],
                                                    img_matches,
                                                    Scalar::all(-1),
                                                    Scalar::all(-1), std::vector<char>(),
                                                    DrawMatchesFlags::DEFAULT);
                                    disp = img_matches;
                                    classification_amount = list_of_object_classifications.size();
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


                                if (seenClassifications.size() == list_of_object_classifications.size() - 1) {
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
        }
        return true;
    }

};

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(ImageSimilarity)
