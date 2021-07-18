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

//imsim
#include <image_similarity/types/similarity_types.h>
#include <pcl/features/rift.h>
#include <pcl/features/intensity_gradient.h>


using namespace uima;


using std::cout;
using std::vector;




class ImageSimilarityPointCloud : public DrawingAnnotator {
private:
    typedef pcl::PointXYZRGBA PointT;

    cv::Mat color, rgb_;
    pcl::PointCloud<PointT>::Ptr bs_cloud_;
    pcl::PointCloud<PointT>::Ptr other_cas_cloud_;
    pcl::PointCloud<pcl::Normal>::Ptr bs_norm, rw_norm;

    pcl::PointCloud<PointT>::Ptr dispCloud;

    vector<cv::DMatch> matches;

    std::set<std::string> list_of_object_classifications;
    std::map<std::string, pcl::PointCloud<PointT>::Ptr> bs_pcl_map, rw_pcl_map;
    std::map<std::string, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> keypoint_pcl_map, bs_keypoint_map, rw_keypoint_map;
    std::map<std::string, pcl::Correspondences> corr_per_object;

    bool save_pcl = false;




    int classification_amount = 0;
    std::map<std::string, std::string> id_to_classname_;

    int iteration = -1;

    bool use_hd_images_ = true;
    std::list<std::string> seenClassifications;
    std::string currentClassification = "";
    bool next = false;
    // Check which object cluster cloud you want to see in the visualizer
    enum {
        ALL_TOGETHER,
        BS_CLOUD_ONLY,
        OTHER_CAS_CLOUD_ONLY,
        KEYPOINTS_CORRESPONDENCES_SINGLE_OBJECT,
        KEYPOINTS_CORRESPONDENCES_SINGLE_OBJECT_WITH_OBJECT
    } dispMode;

    //
    // PARAMETERS
    //
    //
    // This identifier is used to reference the CAS of another AAE that ran before
    // *this* AAE. It is access via rs::CASConsumerContext
    std::string other_cas_id;

public:
    ImageSimilarityPointCloud() : DrawingAnnotator(__func__) {

    }

    TyErrorId initialize(AnnotatorContext &ctx) {
        outInfo("initialize");
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



void calculateSimilarityBetweenClouds(CAS &tcas) {
        for (std::string classification : list_of_object_classifications) {
            if (classification != "unknown") {

                pcl::PointCloud<PointT>::Ptr bs_cloud = bs_pcl_map[classification];
                pcl::PointCloud<PointT>::Ptr rw_cloud = rw_pcl_map[classification];

                if (bs_cloud->points.size() == 0 || rw_cloud->points.size() == 0) {
                    outInfo("Pcl with size 0");
                    continue;
                }

                runPCLPipeline(tcas, classification, bs_cloud, rw_cloud);
                runICP(tcas, classification, bs_cloud, rw_cloud);


            }
        }
    }
    void runICP(CAS &tcas, std::string classification, pcl::PointCloud<PointT>::Ptr bs_cloud, pcl::PointCloud<PointT>::Ptr rw_cloud) {

        pcl::IterativeClosestPoint<PointT, PointT> icp;
        icp.setInputSource(bs_cloud);
        icp.setInputTarget(rw_cloud);
        icp.setMaximumIterations (250);
        icp.setEuclideanFitnessEpsilon(1e-4);

        pcl::PointCloud<PointT>::Ptr this_cas_cluster_cloud_aligned(new pcl::PointCloud<PointT>());
        icp.align(*this_cas_cluster_cloud_aligned);

        outInfo("ICP done. Has converged? " << icp.hasConverged()
                                            << " score: " << icp.getFitnessScore());
        //outInfo("ICP found transform: " << icp.getFinalTransformation());


        rs::SceneCas cas(tcas);
        rs::Scene scene = cas.getScene();

        std::vector<image_similarity::SimilarityObject> imsimobjects;
        scene.annotations.filter(imsimobjects);
        bool contained = false;
        for (image_similarity::SimilarityObject iso: imsimobjects) {
            if (iso.classification.get() == id_to_classname_[classification]) {
                iso.similarityscore_icp(icp.getFitnessScore());
                contained = true;
                break;
            }
        }
        if (contained == false) {
            image_similarity::SimilarityObject imsimObject = rs::create<image_similarity::SimilarityObject>(
                    tcas);
            imsimObject.classification.set(id_to_classname_[classification]);
            imsimObject.similarityscore_icp(icp.getFitnessScore());
            scene.annotations.append(imsimObject);
        }

    }
    void runPCLPipeline(CAS &tcas, std::string classification, pcl::PointCloud<PointT>::Ptr bs_cloud, pcl::PointCloud<PointT>::Ptr rw_cloud) {
        // https://github.com/PointCloudLibrary/pcl/blob/master/examples/keypoints/example_sift_keypoint_estimation.cpp
        pcl::SIFTKeypoint<pcl::PointXYZRGBA, pcl::PointWithScale> sift;
        pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGBA>());
        sift.setSearchMethod(kdtree);
        //sift.setScales(0.003, 8, 8);
//sift.setScales(0.003, 4, 2);

        sift.setScales(0.003, 4, 4);
        sift.setMinimumContrast(  4);
        //sift.setMinimumContrast(  2);
      //  sift.setRadiusSearch(0.01);
        //sift.setRadiusSearch(0.08);
        //sift.setKSearch(20);
        //sift.setMinimumContrast(10.0);
        //sift.setScales(0.01,3,3);
       /* https://github.com/jeffdelmerico/pointcloud_tutorial/blob/master/keypoints/pcl_keypoints.cpp
        const float min_scale = 0.01;
        const int nr_octaves = 3;
        const int nr_octaves_per_scale = 3;
        const float min_contrast = 10.0;
*/
        //aus dem 2d surf auch wenn es hier sift ist^^
        // int nOctaves
        //  The number of a gaussian pyramid octaves that the detector uses. It is set to 4 by default. If you want to get very large features, use the larger value. If you want just small features, decrease it.
        //int nOctaveLayers
        // The number of images within each octave of a gaussian pyramid. It is set to 2 by default.

        //paar keypoints
        //sift.setScales(0.01f, 3, 2);
        //sift.setMinimumContrast(0.0);
        //sift.setRadiusSearch(0.01);

        pcl::PointCloud<pcl::PointWithScale> bs_keypoints;
        pcl::PointCloud<pcl::PointWithScale> rw_keypoints;

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
        outInfo("Keypoints size");
        outInfo(bs_keypoints.points.size());
        outInfo(rw_keypoints.points.size());

        if(save_pcl) {
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
        bs_keypoint_map[classification] = bs_keypoints_ptr;
        rw_keypoint_map[classification] = rw_keypoints_ptr;

        // we need normal estimation here to get thenormals of the pointcloudsnippets and not the normals of the whole cloud.
        //normal esimation ala https://vml.sakura.ne.jp/koeda/PCL/tutorials/html/normal_estimation.html
        pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> ne;

        pcl::PointCloud<pcl::Normal>::Ptr bs_norm(new pcl::PointCloud<pcl::Normal>);
        pcl::PointCloud<pcl::Normal>::Ptr rw_norm(new pcl::PointCloud<pcl::Normal>);

        pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr bs_norm_tree(new pcl::search::KdTree<pcl::PointXYZRGBA>());
        pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr rw_norm_tree(new pcl::search::KdTree<pcl::PointXYZRGBA>());


        ne.setInputCloud(bs_cloud);
        ne.setSearchMethod(bs_norm_tree);
        ne.setRadiusSearch(0.05);
        ne.compute(*bs_norm);

        ne.setInputCloud(rw_cloud);
        ne.setSearchMethod(rw_norm_tree);
        ne.setRadiusSearch(0.05);
        ne.compute(*rw_norm);




        //fpfh from here
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


        //https://docs.ros.org/hydro/api/pcl/html/classpcl_1_1registration_1_1CorrespondenceEstimation.html
        pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> est;
        est.setInputSource(bs_stuff);
        est.setInputTarget(rw_stuff);

        pcl::CorrespondencesPtr corr(new pcl::Correspondences);
        // Determine all reciprocal correspondences
        est.determineReciprocalCorrespondences(*corr);
        outInfo("CorrespondenceEstimation: Found " + std::to_string(corr->size()) + " Correspondences");

        for (auto co: *corr )
        {
           outInfo(co.distance);
        }

        //reject first corrs via distance
        auto rejectdist = pcl::registration::CorrespondenceRejectorDistance();
        rejectdist.setMaximumDistance(15);
        pcl::CorrespondencesPtr corrs_afterDist(new pcl::Correspondences);
        rejectdist.getRemainingCorrespondences(*corr, *corrs_afterDist);
        outInfo(" Dist: " + std::to_string(corrs_afterDist->size()) + " Correspondences");

        for (auto co: *corrs_afterDist )
        {
            outInfo(co.distance);
        }
        //validate rest via ransac
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



        rs::SceneCas cas(tcas);
        rs::Scene scene = cas.getScene();


        std::vector<image_similarity::SimilarityObject> imsimobjects;
        scene.annotations.filter(imsimobjects);
        bool contained = false;
        for (image_similarity::SimilarityObject iso: imsimobjects) {
            if (iso.classification.get() == id_to_classname_[classification]) {
                iso.matches_pcl(corrs.size());
                contained = true;
                break;
            }
        }
        if (contained == false) {
            image_similarity::SimilarityObject imsimObject = rs::create<image_similarity::SimilarityObject>(
                    tcas);
            imsimObject.classification.set(id_to_classname_[classification]);
            imsimObject.matches_pcl(corrs.size());
            scene.annotations.append(imsimObject);
        }
    }



    TyErrorId processWithLock(CAS &tcas, ResultSpecification const &res_spec) {
        outInfo("process start");
        iteration++;
        bs_cloud_ = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
        other_cas_cloud_ = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
        pcl::PointCloud<PointT>::Ptr new_display_cloud(new pcl::PointCloud<PointT>());
        pcl::PointCloud<pcl::Normal>::Ptr new_bs_norm(new pcl::PointCloud<pcl::Normal>);
        pcl::PointCloud<pcl::Normal>::Ptr new_rw_norm(new pcl::PointCloud<pcl::Normal>);
        bs_norm = new_bs_norm;
        rw_norm = new_rw_norm;
        dispCloud = new_display_cloud;

        rs::StopWatch clock;
        rs::SceneCas cas(tcas);
        rs::Scene scene = cas.getScene();


        cas.get(VIEW_CLOUD, *bs_cloud_);
        //cas.get(VIEW_NORMALS, *bs_norm);


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
        other_cas_scene.get(VIEW_CLOUD, *other_cas_cloud_);
        //  other_cas_scene.get(VIEW_NORMALS, *rw_norm);

        if (bs_cloud_->empty() && other_cas_cloud_->empty()) {
            outInfo("other_cas_object doesn't have points. Skipping.");
            return UIMA_ERR_ANNOTATOR_MISSING_INFO;
        }


        //get recognized objects
        //from here rw stuff
        std::vector<rs::Object> rwobjects;
        other_cas_scene.get(VIEW_OBJECTS, rwobjects);
        outInfo("Found " << rwobjects.size() << " rs::Object in RW CAS");

        for (auto rwo: rwobjects) {

            std::string class_name;

            std::vector<rs::Classification> classes;
            rwo.annotations.filter(classes);
            outInfo("Found " << classes.size() << " object annotations in rwo");

            if (classes.size() == 0) {
                outInfo("No classification information for cluster ");
                class_name = "unknown";
            } else {
                class_name = classes[0].classname.get();
            }
            id_to_classname_[rwo.id.get()] = class_name;
            list_of_object_classifications.insert(rwo.id.get());

            //Get point cloud of cluster
            pcl::PointIndicesPtr indices(new pcl::PointIndices());
            rs::conversion::from(static_cast<rs::ReferenceClusterPoints>(rwo.points.get()).indices.get(), *indices);

            pcl::PointCloud<PointT>::Ptr rw_cluster_cloud(new pcl::PointCloud<PointT>);

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
        scene.identifiables.filter(bsobjects);
        outInfo("Found " << bsobjects.size() << " rs::BeliefStateObject in BS CAS");

        for (auto bso: bsobjects) {


            // Fetch the class name from the matching RW cluster
            std::string class_name = id_to_classname_[bso.rsObjectId.get()];

            list_of_object_classifications.insert(bso.rsObjectId.get());

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

            bs_pcl_map[bso.rsObjectId.get()] = bs_cluster_cloud;
            *dispCloud += *bs_cluster_cloud;
        }

        calculateSimilarityBetweenClouds(tcas);
        outInfo("took: " << clock.getTime() << " ms.");
        return UIMA_ERR_NONE;
    }

    void visualizeKeyPointsAndCorr(pcl::visualization::PCLVisualizer &visualizer, std::string classification,
                                   bool withBackground) {
        if (next && (!(std::find(seenClassifications.begin(), seenClassifications.end(),
                                 classification) != seenClassifications.end()) ||
                     currentClassification == "")) {
            currentClassification = classification;
            seenClassifications.push_back(classification);
            next = false;
        }


        if (seenClassifications.size() == list_of_object_classifications.size() - 1) {
            seenClassifications.clear();
        }


        if (currentClassification != "") {
            auto bs = bs_keypoint_map[currentClassification];
            auto rw = rw_keypoint_map[currentClassification];

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

    void drawImageWithLock(cv::Mat &disp) {

        disp = cv::Mat::ones(cv::Size(640, 480), CV_8UC3);
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

        for (std::string classification : list_of_object_classifications) {
            if (classification != "unknown") {
                switch (dispMode) {
                    case ALL_TOGETHER:
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
                }
            }
        }
    }

    bool callbackKey(const int key, const Source source) {
        switch (key) {
            case '1':
                dispMode = ALL_TOGETHER;
                break;
            case '2':
                dispMode = BS_CLOUD_ONLY;
                break;
            case '3':
              dispMode = OTHER_CAS_CLOUD_ONLY;
                break;
            case '4':
                dispMode = KEYPOINTS_CORRESPONDENCES_SINGLE_OBJECT;
                next = true;
                break;
            case '5':
                dispMode = KEYPOINTS_CORRESPONDENCES_SINGLE_OBJECT_WITH_OBJECT;
                next = true;
                break;

        }
        return true;
    }

};

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(ImageSimilarityPointCloud)
