#include <uima/api.hpp>

#include <pcl/point_types.h>


#include <robosherlock/types/all_types.h>
//RS
#include <robosherlock/scene_cas.h>
#include <robosherlock/utils/time.h>
#include <robosherlock/DrawingAnnotator.h>
#include <robosherlock/CASConsumerContext.h>

#include <rs_bs/types/all_types.h>

#include <robosherlock/compare.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <image_similarity/types/all_types.h>
//#include "rs_ue4beliefstate/BeliefStateCommunication.h"

using namespace uima;


class ClusterColorFromBeliefState : public DrawingAnnotator {
private:
    // Annotator parameters from YAML config. Set defaults here.
    int camera_id_ = 0;
    bool use_hd_images_ = false;

    ros::NodeHandle nh_;

    cv::Mat object_, rgb_;
    std::string other_cas_id;

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

    cv::Mat color_mat_;

    bool semantic_label_;

public:
    ClusterColorFromBeliefState() : DrawingAnnotator(__func__), nh_("~"), min_value_color_(60),
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

    TyErrorId initialize(AnnotatorContext &ctx) {
        outInfo("initialize");
        if (ctx.isParameterDefined("camera_id"))
            ctx.extractValue("camera_id", camera_id_);
        if (ctx.isParameterDefined("use_hd_images"))
            ctx.extractValue("use_hd_images", use_hd_images_);
        outInfo("Reading camera data from camera id:" << camera_id_);
        outInfo("Use HD Image streams from Main Cam and Belief State? " << use_hd_images_);

        if (ctx.isParameterDefined("otherCASId")) {
            ctx.extractValue("otherCASId", other_cas_id);
            outInfo("Using AAE/CAS identified by '" << other_cas_id << "' for imagesimilarity");
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

        if(ctx.isParameterDefined("semantic_label"))
        {
            ctx.extractValue("semantic_label", semantic_label_);
        }

        return UIMA_ERR_NONE;
    }

    TyErrorId destroy() {
        outInfo("destroy");
        return UIMA_ERR_NONE;
    }

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
    }


    TyErrorId processWithLock(CAS &tcas, ResultSpecification const &res_spec) {

        outInfo("process start");
        rs::StopWatch clock;
        rs::SceneCas cas(tcas);


        // Color C&P
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
        rs::SceneCas other_cas_scenecas(*other_cas);
        rs::Scene other_cas_scene = other_cas_scenecas.getScene();



        //                                                       TODO WARNING IMPORTANT. This changes the input
        cas.get(VIEW_COLOR_IMAGE, color_mat_, camera_id_);


        bool found = false;
        std::vector<rs_bs::BeliefStateObject> clusters;
        scene.identifiables.filter(clusters);

        cluster_rois_.resize(clusters.size());
        color_ids_.resize(clusters.size(), std::vector<int>(COUNT));
        color_ratios_.resize(clusters.size(), std::vector<float>(COUNT));

        for (size_t idx = 0; idx < clusters.size(); ++idx) {
            rs::ImageROI image_rois = clusters[idx].rois.get();

            //======================= Calculate HSV image ==========================
            cv::Mat rgb, mask;
            cv::Rect roi;
            //                                                       TODO WARNING IMPORTANT. We assume normal-res masks for now.
            // rs::conversion::from(image_rois.roi_hires(), roi);
            //rs::conversion::from(image_rois.mask_hires(), mask);
            rs::conversion::from(image_rois.roi(), roi);
            rs::conversion::from(image_rois.mask(), mask);

            cluster_rois_[idx] = roi;
            outInfo("mop");
            outInfo(roi.size());
            outInfo(mask.size());
            outInfo(color_mat_.size());

            color_mat_(roi).copyTo(rgb, mask(roi));


            cv::Mat hsv, hist;
            cv::cvtColor(rgb, hsv, CV_BGR2HSV_FULL);
            size_t sum;
            std::vector<int> colorCount;
            countColors(hsv, mask(roi), colorCount, sum);

            //======================= Calculate Semantic Color ==========================
            if (found || semantic_label_) {
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
            cv::calcHist(&hsv, 1, channels, mask(roi), hist, 2, histSize, ranges, true, false);

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
            compareWithRWColorHist(color_hist_annotation, scene, other_cas_scene, tcas);
            clusters[idx].annotations.append(color_hist_annotation);
        }


        // Check the color distances - Every cluster should now have a
        // color histogram from the real world data +
        // the simulation
        /*for(size_t idx = 0; idx < clusters.size(); ++idx){
              std::vector<rs::ColorHistogram> colors;
              clusters[idx].annotations.filter(colors);

              outInfo("Object cluster #" << idx << " has " << colors.size() << " color histograms");
              outInfo("  ID is " << clusters[idx].id.get());
              if(colors.size()!=2){
                outError("  Cluster Color executed but still only one color histogram on this cluster. Continue...");
                continue;
              }

              double dist = rs::compare(colors[0], colors[1]);
              outInfo("  matching score " << dist);*/
        /*if(BeliefStateCommunication::belief_changed_in_last_iteration)
        {
          if(dist >= 0.8)
          {
            outInfo("  BeliefState has been changed in last iteration and now there is a mismatch in Cluster idx#" << idx);
            // BeliefStateAccessor::instance()->setMismatchFor(clusters[idx].id.get(), true);
            ros::Duration(5).sleep();
          }else{
            // BeliefStateAccessor::instance()->setMismatchFor(clusters[idx].id.get(), false);
          }
        }*/

        // }


        outInfo("took: " << clock.getTime() << " ms.");
        return UIMA_ERR_NONE;
    }

    void compareWithRWColorHist(rs::ColorHistogram ch, rs::Scene scene, rs::Scene other_cas_scene, CAS &tcas) {

        std::vector<rs::ObjectHypothesis> clusters;
        other_cas_scene.identifiables.filter(clusters);
        outInfo(clusters.size());
        outInfo("colorhistdist");
        std::string class_name = "";
        for(rs::ObjectHypothesis oh : clusters) {
            std::vector<rs::ColorHistogram> colorHistograms;
            oh.annotations.filter(colorHistograms);


            std::vector<rs::Classification> classes;
            oh.annotations.filter(classes);
            class_name = classes[0].classname.get();

            for(rs::ColorHistogram rwHist : colorHistograms) {
                double dist = rs::compare(rwHist, ch);
                outInfo(oh.id.get());
                outInfo(class_name);

                std::vector<image_similarity::SimilarityObject> imsimobjects;
                scene.annotations.filter(imsimobjects);
                bool contained = false;
                for (image_similarity::SimilarityObject iso: imsimobjects) {
                    if (iso.classification.get() == class_name) {
                        iso.colorHist_distance(dist);
                        contained = true;
                        break;
                    }
                }
                if (contained == false) {
                    image_similarity::SimilarityObject imsimObject = rs::create<image_similarity::SimilarityObject>(
                            tcas);
                    imsimObject.classification.set(class_name);
                    imsimObject.colorHist_distance(dist);
                    scene.annotations.append(imsimObject);
                }
            }
        }
    }



    void drawImageWithLock(cv::Mat &disp) {
        disp = color_mat_.clone();
        for (size_t i = 0; i < cluster_rois_.size(); ++i) {
            const cv::Rect &roi = cluster_rois_[i];
            const cv::Size histSize(roi.width, 10);
            const cv::Rect roiHist(roi.x, roi.y + roi.height + 1, histSize.width, histSize.height);
            const std::vector<int> &ids = color_ids_[i];
            const std::vector<float> &ratios = color_ratios_[i];

            cv::rectangle(disp, roi, colors_[ids[0]]);

            cv::Mat hist = disp(roiHist);

            float start = 0;
            for (int r = 0; r < ratios.size(); ++r) {
                float width = (histSize.width * ratios[r]);
                const cv::Rect rect(start + 0.5, 0, width + 0.5, histSize.height);
                start += width;
                cv::rectangle(hist, rect, colors_[ids[r]], CV_FILLED);
            }
        }
    }

};

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(ClusterColorFromBeliefState)
